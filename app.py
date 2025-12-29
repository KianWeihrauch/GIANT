import os
import uuid
import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from pathology_agent_streaming_image_tool_manual_context_claude import PathologyAgentStreamingClaude
from pathology_agent_streaming_image_tool_manual_context import PathologyAgentStreaming
from openslide import OpenSlide
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active conversations
conversations = {}

def get_wsi_images():
    """Get list of WSI images - manually configured"""
    wsi_images = [
        {
            'path': 'TCGA_TEST/test.svs',
            'name': 'TCGA Sample 1',
            'type': 'svs'
        }
    ]
    return wsi_images


def generate_preview(wsi_path, size=(200, 200)):
    """Generate a preview image for a WSI file"""
    slide = OpenSlide(wsi_path)
    
    # Get the best level for thumbnail
    level = slide.level_count - 1
    if slide.level_count > 1:
        # Find a level that gives us roughly the target size
        for i in range(slide.level_count):
            level_size = slide.level_dimensions[i]
            if max(level_size) <= size[0] * 4:  # A bit larger than target for better quality
                level = i
                break
    
    # Get thumbnail from appropriate level
    level_size = slide.level_dimensions[level]
    thumbnail = slide.read_region((0, 0), level, level_size).convert('RGB')
    
    # Resize to target size while maintaining aspect ratio
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    
    # Convert to base64 for web display
    buffer = io.BytesIO()
    thumbnail.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    preview_base64 = base64.b64encode(buffer.read()).decode()
    slide.close()
    
    return f"data:image/jpeg;base64,{preview_base64}"

@app.route('/')
def index():
    """Main chatbot interface"""
    # Get WSI images
    wsi_images = get_wsi_images()
    
    # Generate previews and get dimensions
    for wsi in wsi_images:
        wsi['preview'] = generate_preview(wsi['path'])
        slide = OpenSlide(wsi['path'])
        wsi['dimensions'] = slide.dimensions
        slide.close()
    
    return render_template('chatbot.html', wsi_images=wsi_images)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'message': 'Connected to GIANT'})

@socketio.on('start_conversation')
def handle_start_conversation(data):
    """Start a new conversation with selected WSI"""
    try:
        wsi_file = data.get('wsi_file')
        if not wsi_file:
            emit('error', {'message': 'No WSI file selected'})
            return
        
        # Generate conversation ID
        conv_id = str(uuid.uuid4())
        
        # Create new agent instance with OpenAI gpt-5
        agent = PathologyAgentStreaming(wsi_file, conv_id, max_iterations=20, max_images_in_context=None, model="gpt-5")
        conversations[conv_id] = {
            'agent': agent,
            'wsi_file': wsi_file,
            'messages': []
        }
        
        emit('conversation_started', {
            'conversation_id': conv_id,
            'wsi_file': wsi_file,
            'slide_dimensions': agent.get_slide_dimensions()
        })
        
    except Exception as e:
        emit('error', {'message': f'Failed to start conversation: {str(e)}'})

@socketio.on('send_message')
def handle_message(data):
    """Handle user message and stream AI response"""
    try:
        conv_id = data.get('conversation_id')
        message = data.get('message', 'Please start your examination.')
        
        if conv_id not in conversations:
            emit('error', {'message': 'Conversation not found'})
            return
        
        conversation = conversations[conv_id]
        agent = conversation['agent']
        
        # Add user message to conversation
        conversation['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': None
        })
        
        # Echo user message
        emit('user_message', {'message': message})
        
        # Start streaming AI response
        emit('ai_thinking', {'message': 'AI is analyzing...'})
        
        # Stream the analysis
        for update in agent.analyze_streaming(message, store_intermediate_images=True):
            if update['type'] == 'iteration':
                # Log cost information to console
                step = update.get('step', 0)
                tokens = update.get('tokens', {})
                cost = update.get('cost', {})
                
                # Handle both cost formats (float or dict)
                if isinstance(cost, dict):
                    step_cost = cost.get('step', 0)
                    cumulative_cost = cost.get('cumulative', 0)
                else:
                    step_cost = cost if isinstance(cost, (int, float)) else 0
                    cumulative_cost = 0
                
                # Handle both token formats
                if isinstance(tokens, dict):
                    input_tokens = tokens.get('input', tokens.get('prompt', 0))
                    output_tokens = tokens.get('output', tokens.get('completion', 0))
                else:
                    input_tokens = 0
                    output_tokens = 0
                
                print(f"[Step {step}] Tokens: {input_tokens:,} in / {output_tokens:,} out | "
                      f"Cost: ${step_cost:.4f} this step, ${cumulative_cost:.4f} cumulative")
                emit('iteration_update', update)
            elif update['type'] == 'note':
                emit('note_update', update)
            elif update['type'] == 'final':
                # Log final cost summary
                metadata = update.get('metadata', {})
                tokens = metadata.get('tokens', {})
                total_cost = metadata.get('total_cost', 0.0)
                print(f"\n{'='*70}")
                print(f"FINAL ANALYSIS COMPLETE")
                print(f"{'='*70}")
                print(f"Total Iterations: {metadata.get('total_iterations', 0)}")
                print(f"Total Input Tokens: {tokens.get('input', 0):,}")
                print(f"Total Output Tokens: {tokens.get('output', 0):,}")
                print(f"Total Tokens: {tokens.get('total', 0):,}")
                print(f"TOTAL COST: ${total_cost:.4f}")
                print(f"{'='*70}\n")
                
                emit('final_analysis', update)
                # Add final message to conversation
                conversation['messages'].append({
                    'role': 'assistant',
                    'content': update['analysis'],
                    'timestamp': None,
                    'metadata': metadata
                })
            elif update['type'] == 'error':
                emit('error', update)
                break
        
    except Exception as e:
        emit('error', {'message': f'Analysis failed: {str(e)}'})

@socketio.on('reset_conversation')
def handle_reset_conversation(data):
    """Reset the conversation while keeping the same WSI"""
    try:
        conv_id = data.get('conversation_id')
        
        if conv_id not in conversations:
            emit('error', {'message': 'Conversation not found'})
            return
        
        conversation = conversations[conv_id]
        wsi_file = conversation['wsi_file']
        
        # Create new agent instance (resets the conversation) with OpenAI gpt-5
        new_agent = PathologyAgentStreaming(wsi_file, conv_id, max_iterations=30, max_images_in_context=5, model="gpt-5")
        conversation['agent'] = new_agent
        conversation['messages'] = []
        
        emit('conversation_reset', {
            'conversation_id': conv_id,
            'message': 'Conversation reset. You can start a new analysis.'
        })
        
    except Exception as e:
        emit('error', {'message': f'Failed to reset conversation: {str(e)}'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    socketio.run(app, debug=True, port=3010) 