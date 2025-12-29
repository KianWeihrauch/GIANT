import base64, io, json, math, os
from configparser import ConfigParser
from util import encode_image, crop_to_image_with_guides
from openslide import OpenSlide
from anthropic import Anthropic
import shutil
import time

SYSTEM_PROMPT_MULTI_STEP = """You are a meticulous pathology agent with expertise in histopathological analysis. Your goal is to achieve the most comprehensive understanding possible of the tissue slide.

COORDINATE SYSTEM: The full slide resolution is {w0} × {h0} pixels. All coordinates must reference this full resolution. The position guides on each image show full-resolution pixel coordinates.

ANALYSIS STRATEGY: You have {steps_remaining} total steps available. Use up to {analysis_steps} steps for intensive slide examination using slide_zoom. Reserve your final step for reporting.
{context_limitation}
TOOL USAGE RESTRICTION: You can only make ONE tool call per turn. If you invoke multiple tools in a single response, only the FIRST tool will be executed and the rest will be ignored. Plan your examination strategy accordingly - examine one region at a time.

EXAMINATION PHASE (Steps 1-{analysis_steps}):
- CRITICAL: Use as many steps as possible to thoroughly analyze the slide
- Examine multiple regions at different magnifications systematically
- Look for tissue architecture patterns, cellular morphology details, and any pathological changes
- Zoom into areas of interest multiple times at different scales
- Do NOT rush - spend time understanding each region before moving on
- Continue examining until you have achieved comprehensive understanding OR reached step {analysis_steps}
{note_tool_guidance}
REPORTING PHASE (Final Step):
Structure your final response with these exact XML tags:

<report>
Provide a detailed pathological report including:
- Overall tissue architecture and structural organization
- Cellular morphology and characteristics observed
- Any abnormalities, lesions, or pathological changes identified
- Description of specific regions examined and key findings
- Diagnostic reasoning based on your observations{note_reminder}
</report>

<response>
Provide your specific answer to the user's question based on your comprehensive analysis.
</response>

REMEMBER: Quality over speed - use all available analysis steps to achieve the most thorough understanding possible before providing your final report and answer.
"""

SYSTEM_PROMPT_SINGLE_STEP = """You are a meticulous pathology agent with expertise in histopathological analysis. Your goal is to achieve the most comprehensive understanding possible of the tissue slide.

Structure your final response with these exact XML tags:

<report>
Provide a detailed pathological report including:
- Overall tissue architecture and structural organization
- Cellular morphology and characteristics observed
- Any abnormalities, lesions, or pathological changes identified
- Description of specific regions examined and key findings
- Your diagnostic reasoning based on your observations
</report>

<response>
Provide your specific answer to the user's question based on your comprehensive analysis.
</response>
"""

class PathologyAgentStreamingClaude:
    def __init__(self, wsi_file, session_id, max_iterations=15, model="claude-sonnet-4-5-20250929", max_images_in_context=None, enable_note_tool=False, target_size_long_size=500):
        # Config
        config = ConfigParser()
        config.read("config.ini")
        api_key = config.get("main", "ANTHROPIC_API_KEY")

        self.client = Anthropic(api_key=api_key)

        self.MODEL_ID = model
        self.MAX_ITERS = max_iterations
        self.max_images_in_context = max_images_in_context  # If None, keep all images
        self.enable_note_tool = enable_note_tool
        self.target_size_long_size = target_size_long_size  # Max resolution for image long side
        
        # Session and file setup
        self.session_id = session_id
        self.wsi_file = wsi_file
        self.slide = OpenSlide(wsi_file)
        self.debug_dir = f"debug/{session_id}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Conversation state
        self.conversation = []  # Store the entire conversation history
        self.iteration_count = 0
        self.raw_responses = []  # Store raw response dictionaries for each iteration
        self.notes = []  # Store important findings/notes
        
        # Cost tracking
        self.cumulative_cost = 0.0
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        
        # Pricing for Claude Sonnet 4.5 (per million tokens)
        # Update these if using a different model
        self.INPUT_COST_PER_MILLION = 3.00
        self.OUTPUT_COST_PER_MILLION = 15.00
        
        # Tool schema (Claude format)
        self.tools = [{
            "name": "slide_zoom",
            "description": "Crop a rectangular region of the whole-slide image. Coordinate system: x-axis runs left to right, y-axis runs top to bottom. x0,y0 is the top-left corner of the region.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x0": {"type": "integer", "description": "Left edge x-coordinate (x-axis: left to right)"},
                    "y0": {"type": "integer", "description": "Top edge y-coordinate (y-axis: top to bottom)"},
                    "w": {"type": "integer", "description": "Width of the region in pixels"},
                    "h": {"type": "integer", "description": "Height of the region in pixels"}
                },
                "required": ["x0", "y0", "w", "h"]
            }
        }]
        
        # Add note tool if enabled
        if self.enable_note_tool:
            self.tools.append({
                "name": "note",
                "description": "Record important findings, observations, or anything you need to remember for your final report. Use this to document crucial features, abnormalities, suspicious regions, cancerous tissue, specific locations, or any key pathological observations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string", "description": "The observation, finding, or note to record for later reference"}
                    },
                    "required": ["observation"]
                }
            })

        # Initialize with system message if this is the first conversation
        if not self.conversation:
            self._initialize_conversation()

    def get_slide_dimensions(self):
        """Get slide dimensions"""
        w0, h0 = self.slide.dimensions
        return {"width": w0, "height": h0}

    def _initialize_conversation(self):
        """Initialize conversation with system message"""
        w0, h0 = self.slide.dimensions
        
        # Choose appropriate prompt based on max_iterations
        if self.MAX_ITERS == 1:
            # Single step - no tool use, just analyze overview
            prompt_content = SYSTEM_PROMPT_SINGLE_STEP.format(w0=w0, h0=h0)
        else:
            # Multi-step - use tools for detailed analysis
            # Build context limitation text if sliding window is enabled
            context_limitation = ""
            if self.max_images_in_context is not None:
                context_limitation = f"\nCONTEXT WINDOW MANAGEMENT: Due to context limitations, you can only see the overview image and the {self.max_images_in_context - 1} most recent zoomed regions at any time. Older images will be removed from your view but their text descriptions remain."
            
            # Build note tool guidance if enabled
            note_tool_guidance = ""
            note_reminder = ""
            if self.enable_note_tool:
                note_tool_guidance = "\nIMPORTANT FINDINGS TOOL: Use the 'note' tool to record crucial observations, abnormalities, suspicious features, or any significant findings as you discover them. This helps you remember key details for your final report, especially when examining many regions."
                if self.max_images_in_context is not None:
                    note_tool_guidance += " Since older images will be removed from your context, the note tool is CRITICAL for preserving important discoveries throughout your examination."
                note_reminder = "\n- Reference any notes you recorded during examination"
            
            prompt_content = SYSTEM_PROMPT_MULTI_STEP.format(
                w0=w0, 
                h0=h0, 
                steps_remaining=self.MAX_ITERS,
                analysis_steps=self.MAX_ITERS - 1,
                context_limitation=context_limitation,
                note_tool_guidance=note_tool_guidance,
                note_reminder=note_reminder
            )
        
        # Store system prompt separately for Claude (not in conversation)
        self.system_prompt = prompt_content

    def _calculate_cost(self, input_tokens, output_tokens):
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost
    
    def _print_token_usage(self, response, step, step_cost):
        """Print token usage and cost from Claude response object"""
        try:
            # Try to access usage from response object
            usage = response.usage
            
            if usage:
                # Extract token counts
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                total_tokens = input_tokens + output_tokens

                print(f"\n{'='*70}")
                print(f"STEP {step} - TOKEN USAGE & COST")
                print(f"{'='*70}")
                print(f"Input tokens:      {input_tokens:>10,}  (${input_tokens / 1_000_000 * self.INPUT_COST_PER_MILLION:>8.4f})")
                print(f"Output tokens:     {output_tokens:>10,}  (${output_tokens / 1_000_000 * self.OUTPUT_COST_PER_MILLION:>8.4f})")
                print(f"{'-'*70}")
                print(f"Step total:        {total_tokens:>10,}  (${step_cost:>8.4f})")
                print(f"{'='*70}")
                print(f"CUMULATIVE TOTAL:  {self.cumulative_input_tokens + self.cumulative_output_tokens:>10,}  (${self.cumulative_cost:>8.4f})")
                print(f"{'='*70}\n")
            else:
                print(f"[STEP {step}] No usage data found in response")
                
        except Exception as e:
            print(f"[STEP {step}] Error extracting token usage: {e}")

    def _prune_images_from_context(self):
        """
        Prune older images from conversation to prevent context window depletion.
        Keeps the overview image (in first user message) and the N most recent images 
        from user messages (tool results). Only removes images, preserves text.
        """
        if self.max_images_in_context is None:
            return  # No pruning if max_images_in_context is None
        
        # Find all user messages with images and track their indices
        user_message_indices = []
        overview_index = None
        
        for idx, item in enumerate(self.conversation):
            # Find user messages with images
            if item.get("role") == "user":
                if isinstance(item.get("content"), list):
                    has_image = any(c.get("type") == "image" for c in item["content"])
                    if has_image:
                        if overview_index is None:
                            overview_index = idx  # First user message with image is overview
                        else:
                            user_message_indices.append(idx)
        
        # Calculate total images
        total_images = len(user_message_indices)
        if overview_index is not None:
            total_images += 1
        
        # Check if pruning is needed
        if total_images <= self.max_images_in_context:
            return  # No pruning needed
        
        # Determine how many user message images to keep (max_images - 1 for overview)
        images_to_keep = self.max_images_in_context - 1 if overview_index is not None else self.max_images_in_context
        
        # Keep the most recent N user message images
        if len(user_message_indices) > images_to_keep:
            indices_to_prune = user_message_indices[:-images_to_keep]  # Keep last N, prune older ones
            
            # Remove images from the user messages (but keep text)
            for idx in indices_to_prune:
                item = self.conversation[idx]
                if isinstance(item.get("content"), list):
                    # Filter out image items, keep text items
                    item["content"] = [
                        content for content in item["content"]
                        if content.get("type") != "image"
                    ]
            
            #print(f"[PRUNING] Removed {len(indices_to_prune)} older images from context")

    # Wrapper of crop_image_to_guides that also saves a local version for the web app
    def crop_to_data_url(self, x0, y0, w, h, step, save=False):
        region_with_guides = crop_to_image_with_guides(self.slide, x0, y0, w, h, target_max=self.target_size_long_size)

        buf = io.BytesIO()
        region_with_guides.save(buf, format="JPEG")
        buf.seek(0)

        static_fn = None
        if save:
            fn = f"{self.debug_dir}/step_{step}.jpg"
            region_with_guides.save(fn, "JPEG")

            # Copy to static for web access
            static_dir = f"static/{self.session_id}"
            os.makedirs(static_dir, exist_ok=True)
            static_fn = f"{static_dir}/step_{step}.jpg"
            shutil.copy2(fn, static_fn)

        # Return base64 string (without data URL prefix for Claude) and static path
        return base64.b64encode(buf.read()).decode(), static_fn

    def analyze_streaming(self, user_message, store_intermediate_images=False):
        """Stream analysis updates in real-time"""
        try:
            # If this is the first message, add overview
            w0, h0 = self.slide.dimensions
            overview_base64, overview_static_path = self.crop_to_data_url(0, 0, w0, h0, step=0, save=store_intermediate_images)

            # Add user message with overview to conversation (Claude format)
            self.conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": overview_base64
                        }
                    }
                ]
            })

            # Yield overview
            yield {
                "type": "iteration",
                "step": 0,
                "iteration_type": "overview",
                "image_url": f"/{overview_static_path}",
                "coordinates": {"x0": 0, "y0": 0, "w": w0, "h": h0},
                "message": f"Starting analysis of slide ({w0} × {h0} pixels)",
                "tokens": {"input": 0, "output": 0, "total": 0},
                "cost": {"step": 0.0, "cumulative": 0.0}
            }

            # Analysis loop
            for step in range(1, self.MAX_ITERS + 1):
                self.iteration_count += 1
                
                # Determine if tools should be available (only if max_iterations > 1)
                tools_to_use = self.tools if self.MAX_ITERS > 1 else None
                
                # Send request to Claude with full conversation history (using streaming)
                with self.client.messages.stream(
                    model=self.MODEL_ID,
                    max_tokens=30000,
                    system=self.system_prompt,
                    messages=self.conversation,
                    tools=tools_to_use,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                ) as stream:
                    # Consume the stream to get the final response
                    response = stream.get_final_message()

                # Calculate and track token usage from response
                step_cost = self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
                self.cumulative_input_tokens += response.usage.input_tokens
                self.cumulative_output_tokens += response.usage.output_tokens
                self.cumulative_cost += step_cost
                # Token usage printing disabled
                # self._print_token_usage(response, step, step_cost)

                # Store raw response dictionary for this iteration
                self.raw_responses.append({
                    "iteration": step,
                    "response": response.model_dump()
                })

                # Add assistant response to conversation
                self.conversation.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Check stop reason
                if response.stop_reason == "tool_use":
                    # Process tool calls - but only execute the FIRST one
                    tool_results = []
                    tool_executed = False
                    
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
                            tool_use_id = content_block.id
                            
                            # If we've already executed a tool, reject subsequent ones
                            if tool_executed:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": "Error: Only one tool call per turn is allowed. This tool call was ignored. Please make one tool call at a time."
                                })
                                continue
                            
                            try:
                                # Handle note tool
                                if tool_name == 'note':
                                    observation = tool_input.get('observation', '')
                                    
                                    # Store the note
                                    note_entry = {
                                        'step': step,
                                        'observation': observation
                                    }
                                    self.notes.append(note_entry)
                                    
                                    # Create tool result
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": f"Note recorded: {observation}"
                                    })
                                    
                                    # Yield note update
                                    yield {
                                        "type": "note",
                                        "step": step,
                                        "observation": observation,
                                        "message": f"Note recorded: {observation}"
                                    }
                                    tool_executed = True
                                    continue
                                
                                # Handle slide_zoom tool
                                x0 = tool_input['x0']
                                y0 = tool_input['y0']
                                w = tool_input['w']
                                h = tool_input['h']
                                
                                # Crop the region
                                region_base64, static_path = self.crop_to_data_url(x0, y0, w, h, step=step, save=store_intermediate_images)

                                # Extract any text and thinking content from the response for reasoning
                                ai_reasoning = ""
                                for content_block in response.content:
                                    if content_block.type == "text":
                                        ai_reasoning += content_block.text + " "
                                    elif content_block.type == "thinking":
                                        ai_reasoning += content_block.thinking + " "
                                ai_reasoning = ai_reasoning.strip() or "Analyzing region..."

                                # Yield iteration update with cost information
                                yield {
                                    "type": "iteration",
                                    "step": step,
                                    "iteration_type": "zoom",
                                    "image_url": f"/{static_path}",
                                    "coordinates": {"x0": x0, "y0": y0, "w": w, "h": h},
                                    "message": f"Examining region at top-left ({x0}, {y0}) - {w} × {h} pixels",
                                    "ai_reasoning": ai_reasoning,
                                    "tokens": {
                                        "input": response.usage.input_tokens,
                                        "output": response.usage.output_tokens,
                                        "total": response.usage.input_tokens + response.usage.output_tokens
                                    },
                                    "cost": {
                                        "step": step_cost,
                                        "cumulative": self.cumulative_cost
                                    }
                                }

                                # Calculate steps remaining
                                steps_remaining = self.MAX_ITERS - step - 1
                                
                                # Get full slide dimensions for context message
                                w0_full, h0_full = self.slide.dimensions
                                
                                # Build reminder text
                                reminder_text = (
                                    f"Here is the region with top-left corner at ({x0},{y0}) and size {w}×{h} pixels. "
                                    f"REMINDER: Full slide resolution is {w0_full} × {h0_full} pixels - all coordinates must reference this full resolution. "
                                    f"You have {steps_remaining} remaining function calls."
                                )
                                
                                # Add notes summary if any notes exist and note tool is enabled
                                if self.enable_note_tool and self.notes:
                                    reminder_text += f"\n\nYour recorded notes so far ({len(self.notes)} total):\n"
                                    for i, note in enumerate(self.notes, 1):
                                        reminder_text += f"{i}. {note['observation']}\n"
                                
                                # Create tool result with image
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": reminder_text
                                        },
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": region_base64
                                            }
                                        }
                                    ]
                                })
                                tool_executed = True
                                
                            except Exception as e:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"Error: {str(e)}"
                                })
                                tool_executed = True
                    
                    # Add tool results to conversation as a user message
                    if tool_results:
                        self.conversation.append({
                            "role": "user",
                            "content": tool_results
                        })
                        
                        # Prune images if needed to manage context window
                        self._prune_images_from_context()
                    
                    continue

                # Handle final response (no more tool calls)
                elif response.stop_reason == "end_turn":
                    # Extract final text content
                    final_content = ""
                    for content_block in response.content:
                        if content_block.type == "text":
                            final_content += content_block.text
                    
                    yield {
                        "type": "final",
                        "analysis": final_content,
                        "metadata": {
                            "total_iterations": self.iteration_count,
                            "tokens": {
                                "input": self.cumulative_input_tokens,
                                "output": self.cumulative_output_tokens,
                                "total": self.cumulative_input_tokens + self.cumulative_output_tokens
                            },
                            "total_cost": self.cumulative_cost,
                            "slide_file": self.wsi_file,
                            "notes": self.notes
                        },
                        "raw_responses": self.raw_responses,
                        "max_iterations_reached": False
                    }
                    return
                
                # Handle max_tokens stop reason
                elif response.stop_reason == "max_tokens":
                    # Continue to next iteration
                    continue

            # If we hit max iterations
            yield {
                "type": "final",
                "analysis": "Analysis completed after maximum iterations. The examination covered multiple regions of the slide.",
                "metadata": {
                    "total_iterations": self.iteration_count,
                    "tokens": {
                        "input": self.cumulative_input_tokens,
                        "output": self.cumulative_output_tokens,
                        "total": self.cumulative_input_tokens + self.cumulative_output_tokens
                    },
                    "total_cost": self.cumulative_cost,
                    "slide_file": self.wsi_file,
                    "note": "Reached maximum iteration limit",
                    "notes": self.notes
                },
                "raw_responses": self.raw_responses,
                "max_iterations_reached": True
            }

        except Exception as e:
            yield {
                "type": "error",
                "message": f"Analysis error: {str(e)}",
                "notes": self.notes
            } 