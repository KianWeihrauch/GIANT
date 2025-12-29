import base64, io, json, math, os
from configparser import ConfigParser
from util import encode_image, crop_to_image_with_guides
from openslide import OpenSlide
from openai import OpenAI, AzureOpenAI
import shutil
import time

SYSTEM_PROMPT_MULTI_STEP = """You are a meticulous pathology agent with expertise in histopathological analysis. Your goal is to achieve the most comprehensive understanding possible of the tissue slide.

COORDINATE SYSTEM: The full slide resolution is {w0} × {h0} pixels. All coordinates must reference this full resolution. The position guides on each image show full-resolution pixel coordinates.

ANALYSIS STRATEGY: You have {steps_remaining} total steps available. Use up to {analysis_steps} steps for intensive slide examination using slide_zoom. Reserve your final step for reporting.
{context_limitation}
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

class PathologyAgentStreaming:
    def __init__(self, wsi_file, session_id, max_iterations=15, model="gpt-5", max_images_in_context=None, enable_note_tool=False, target_size_long_size=1000):
        # Config

        config = ConfigParser()
        config.read("config.ini")
        
        # Determine which client to use based on model
        # Regular OpenAI for: gpt-4o, gpt-5, o1, o3, o1-mini, o3-mini, etc.
        # Azure for: Azure-specific deployments
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-5", "o1", "o3", "o1-mini", "o3-mini", "o1-preview", "o3-mini"]
        
        if model in openai_models:
            api_key = config.get("main", "OPENAI_KEY_TB")
            self.client = OpenAI(api_key=api_key)
            self.is_azure = False
        else:
            api_key = config.get("main", "AZURE_OPENAI_API_KEY")
            endpoint = config.get("main", "AZURE_OPENAI_ENDPOINT")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2025-03-01-preview",
                azure_endpoint=endpoint
            )
            self.is_azure = True

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
        
        # Tool schema
        self.tools = [{
            "type": "function",
            "name": "slide_zoom",
            "description": "Crop a rectangular region of the whole-slide image. Coordinate system: x-axis runs left to right, y-axis runs top to bottom. x0,y0 is the top-left corner of the region.",
            "parameters": {
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
                "type": "function",
                "name": "note",
                "description": "Record important findings, observations, or anything you need to remember for your final report. Use this to document crucial features, abnormalities, suspicious regions, cancerous tissue, specific locations, or any key pathological observations.",
                "parameters": {
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
        
        self.conversation.append({
            "role": "system",
            "type": "message",
            "content": prompt_content
        })

    def _print_token_usage(self, response, step):
        """Print token usage from OpenAI response object"""
        try:
            # Try to access usage from response object
            usage = None
            usage = response.usage
            
            if usage:
                # Extract token counts
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                reasoning_tokens = usage.output_tokens_details.reasoning_tokens

                print(f"\n{'='*60}")
                print(f"TOKEN USAGE - Step {step}")
                print(f"{'='*60}")
                print(f"Input tokens:     {input_tokens:,}")
                print(f"Output tokens: {output_tokens:,}")
                print(f"Reasoning tokens (part of output):      {reasoning_tokens:,}")
                print(f"{'='*60}\n")
            else:
                print(f"[STEP {step}] No usage data found in response")
                
        except Exception as e:
            print(f"[STEP {step}] Error extracting token usage: {e}")

    def _prune_images_from_context(self):
        """
        Prune older images from conversation to prevent context window depletion.
        Keeps the overview image (in first user message) and the N most recent images 
        from function_call_output items. Only removes images, preserves text and reasoning.
        """
        if self.max_images_in_context is None:
            return  # No pruning if max_images_in_context is None
        
        # Find all items with images and track their indices
        function_output_indices = []
        overview_index = None
        
        for idx, item in enumerate(self.conversation):
            # Find the overview image (first user message with image)
            if item.get("type") == "message" and item.get("role") == "user" and overview_index is None:
                if isinstance(item.get("content"), list):
                    for content_item in item["content"]:
                        if content_item.get("type") == "input_image":
                            overview_index = idx
                            break
            
            # Find function_call_output items with images
            elif item.get("type") == "function_call_output":
                output = item.get("output")
                if isinstance(output, list):
                    has_image = any(o.get("type") == "input_image" for o in output)
                    if has_image:
                        function_output_indices.append(idx)
        
        # Calculate total images
        total_images = len(function_output_indices)
        if overview_index is not None:
            total_images += 1
        
        # Check if pruning is needed
        if total_images <= self.max_images_in_context:
            return  # No pruning needed
        
        # Determine how many function output images to keep (max_images - 1 for overview)
        images_to_keep = self.max_images_in_context - 1 if overview_index is not None else self.max_images_in_context
        
        # Keep the most recent N function output images
        if len(function_output_indices) > images_to_keep:
            indices_to_prune = function_output_indices[:-images_to_keep]  # Keep last N, prune older ones
            
            # Remove images from the function_call_output items (but keep text)
            for idx in indices_to_prune:
                item = self.conversation[idx]
                if isinstance(item.get("output"), list):
                    # Filter out image items, keep text items
                    item["output"] = [
                        content for content in item["output"]
                        if content.get("type") != "input_image"
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

        return f"data:image/jpeg;base64,{base64.b64encode(buf.read()).decode()}", static_fn

    def analyze_streaming(self, user_message, store_intermediate_images=False):
        """Stream analysis updates in real-timetime"""
        try:
            # If this is the first message, add overview
            w0, h0 = self.slide.dimensions
            overview_url, overview_static_path = self.crop_to_data_url(0, 0, w0, h0, step=0, save=store_intermediate_images)

            # Add user message with overview to conversation
            self.conversation.append({
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": user_message},
                    {"type": "input_image", "image_url": overview_url},
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
                "tokens": {"prompt": 0, "completion": 0},
                "cost": 0.0
            }

            # Analysis loop
            for step in range(1, self.MAX_ITERS + 1):
                self.iteration_count += 1
                
                # Determine if tools should be available (only if max_iterations > 1)
                tools_to_use = self.tools if self.MAX_ITERS > 1 else None
                
                # Determine if model supports reasoning parameter
                # Reasoning models: gpt-5, o1, o3, and their variants
                reasoning_models = ["gpt-5", "o1", "o3", "o1-mini", "o3-mini", "o1-preview"]
                supports_reasoning = self.MODEL_ID in reasoning_models
                
                # Build request parameters
                request_params = {
                    "input": self.conversation,
                    "model": self.MODEL_ID,
                    "tools": tools_to_use
                }
                
                # Only add reasoning parameter for reasoning models
                if supports_reasoning:
                    request_params["reasoning"] = {"effort": "high", "summary": "detailed"}
                
                # Send request to OpenAI using responses API with full conversation history
                response = self.client.responses.create(**request_params)

                # Print token usage from response
                #self._print_token_usage(response, step)

                # Store raw response dictionary for this iteration
                response_dict = response.to_dict()
                self.raw_responses.append({
                    "iteration": step,
                    "response": response_dict
                })

                # Append all response output items to conversation (includes reasoning, messages, function calls)
                for output_item in response.output:
                    self.conversation.append(output_item.to_dict())

                function_calls = [rx.to_dict() for rx in response.output if rx.type == 'function_call']
                messages = [rx.to_dict() for rx in response.output if rx.type == 'message']

                # Handle function calls
                if len(function_calls) > 0:
                    # Process function calls and create outputs
                    for response_item in response.output:
                        if response_item.type == 'function_call':
                            try:
                                args = json.loads(response_item.arguments)
                                
                                # Handle note tool
                                if response_item.name == 'note':
                                    observation = args.get('observation', '')
                                    
                                    # Store the note
                                    note_entry = {
                                        'step': step,
                                        'observation': observation
                                    }
                                    self.notes.append(note_entry)
                                    
                                    # Create function output
                                    function_output = {
                                        "type": "function_call_output",
                                        "call_id": response_item.call_id,
                                        "output": f"Note recorded: {observation}"
                                    }
                                    self.conversation.append(function_output)
                                    
                                    # Yield note update
                                    yield {
                                        "type": "note",
                                        "step": step,
                                        "observation": observation,
                                        "message": f"Note recorded: {observation}"
                                    }
                                    continue
                                
                                # Handle slide_zoom tool
                                # Crop the region
                                data_url, static_path = self.crop_to_data_url(**args, step=step, save=store_intermediate_images)

                                # Get AI reasoning from response
                                ai_reasoning_items = []
                                try:
                                    # Look for reasoning in the output array
                                    if hasattr(response, 'output') and response.output:
                                        for output_item in response.output:
                                            if hasattr(output_item, 'type') and output_item.type == 'reasoning':
                                                if hasattr(output_item, 'summary') and output_item.summary:
                                                    for summary_item in output_item.summary:
                                                        if hasattr(summary_item, 'text'):
                                                            ai_reasoning_items.append(summary_item.text)
                                                break
                                    ai_reasoning = ' '.join(ai_reasoning_items)
                                    
                                    # Fallback to other methods if reasoning not found in output
                                    if not ai_reasoning and hasattr(response, 'output_text') and response.output_text:
                                        ai_reasoning = response.output_text
                                    
                                    # Final fallback
                                    if not ai_reasoning:
                                        ai_reasoning = 'Analyzing region...'
                                        
                                except Exception as e:
                                    print(f"Error extracting reasoning: {e}")
                                    ai_reasoning = 'Reasoning extraction failed'

                                # Yield iteration update
                                yield {
                                    "type": "iteration",
                                    "step": step,
                                    "iteration_type": "zoom",
                                    "image_url": f"/{static_path}",
                                    "coordinates": args,
                                    "message": f"Examining region at top-left ({args['x0']}, {args['y0']}) - {args['w']} × {args['h']} pixels",
                                    "ai_reasoning": ai_reasoning,
                                    "tokens": {"prompt": 0, "completion": 0},
                                    "cost": 0.0
                                }

                                # have to subtract one since the last step cannot be a function call
                                steps_remaining = self.MAX_ITERS - step - 1 
                                
                                # Get full slide dimensions for context message
                                w0_full, h0_full = self.slide.dimensions
                                
                                # Build reminder text
                                reminder_text = (
                                    f"Here is the region with top-left corner at ({args['x0']},{args['y0']}) and size {args['w']}×{args['h']} pixels. "
                                    f"REMINDER: Full slide resolution is {w0_full} × {h0_full} pixels - all coordinates must reference this full resolution. "
                                    f"You have {steps_remaining} remaining function calls."
                                )
                                
                                # Add notes summary if any notes exist and note tool is enabled
                                if self.enable_note_tool and self.notes:
                                    reminder_text += f"\n\nYour recorded notes so far ({len(self.notes)} total):\n"
                                    for i, note in enumerate(self.notes, 1):
                                        reminder_text += f"{i}. {note['observation']}\n"
                                
                                # Create function output and append to conversation
                                function_output = {
                                    "type": "function_call_output",
                                    "call_id": response_item.call_id,
                                    "output": [
                                        {
                                            "type": "input_text",
                                            "text": reminder_text
                                        },
                                        {
                                            "type": "input_image",
                                            "detail": "high",
                                            "image_url": data_url
                                        }
                                    ]
                                }
                                self.conversation.append(function_output)
                                
                                # Prune images if needed to manage context window
                                self._prune_images_from_context()
                                
                            except Exception as e:
                                function_output = {
                                    "type": "function_call_output",
                                    "call_id": response_item.call_id,
                                    "output": f"Error: {str(e)}"
                                }
                                self.conversation.append(function_output)

                    continue

                # Handle final messages (no more function calls)
                if len(messages) > 0:
                    # Analysis complete - use response.output_text from responses API
                    final_content = response.output_text if hasattr(response, 'output_text') and response.output_text else 'Analysis completed.'
                    
                    yield {
                        "type": "final",
                        "analysis": final_content,
                        "metadata": {
                            "total_iterations": self.iteration_count,
                            "total_tokens": 0,
                            "total_cost": 0.0,
                            "slide_file": self.wsi_file,
                            "notes": self.notes
                        },
                        "raw_responses": self.raw_responses,
                        "max_iterations_reached": False
                    }
                    return

                # If no function calls and no messages, something went wrong
                if len(function_calls) == 0 and len(messages) == 0:
                    yield {
                        "type": "final",
                        "analysis": "Analysis completed. No further actions requested.",
                        "metadata": {
                            "total_iterations": self.iteration_count,
                            "total_tokens": 0,
                            "total_cost": 0.0,
                            "slide_file": self.wsi_file,
                            "notes": self.notes
                        },
                        "raw_responses": self.raw_responses,
                        "max_iterations_reached": False
                    }
                    return

            # If we hit max iterations
            yield {
                "type": "final",
                "analysis": "Analysis completed after maximum iterations. The examination covered multiple regions of the slide.",
                "metadata": {
                    "total_iterations": self.iteration_count,
                    "total_tokens": 0,
                    "total_cost": 0.0,
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