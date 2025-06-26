# Freeze-Omni Enhanced Demo

This enhanced demo provides comprehensive real-time visualization of the Freeze-Omni speech-to-speech dialogue system's internal states and processes.

## Features

### 1. Basic Functionality
- All features from the original demo
- Speech-to-speech dialogue interaction
- Real-time audio waveform visualization
- System prompt configuration

### 2. Real-time State Monitoring
The enhanced demo displays the current system status:

- **VAD State**: Shows whether Voice Activity Detection is currently active (user speaking detected)
- **Dialog State**: Displays the current dialogue state machine state:
  - `dialog_sl`: Start Listen - Beginning to listen for user input
  - `dialog_cl`: Continue Listen - Continuing to listen (user still speaking)
  - `dialog_el`: End Listen - Stopped listening (user speech ended, but no response generated)
  - `dialog_ss`: Start Speak - Beginning to generate and speak response
  - `dialog_cs`: Continue Speak - Continuing to generate and speak response
- **Generation Status**: Shows whether the system is currently generating a response

### 3. Timeline Visualizations (10-second windows)

#### VAD State Timeline
- Green bars: VAD is active (user speech detected)
- Light green/empty: VAD is inactive
- Updates every 100ms for smooth visualization

#### Generation Timeline
- Yellow bars: System is generating a response
- Empty: System is not generating
- Shows overlap between user speech and system generation (interruption scenarios)

#### Dialog State Timeline
- Five-strip visualization showing the dialogue state over time
- Each state has its own color and position:
  - **SL (Start Listen)**: Light green, top strip
  - **CL (Continue Listen)**: Blue, second strip  
  - **EL (End Listen)**: Light red, middle strip
  - **SS (Start Speak)**: Yellow, fourth strip
  - **CS (Continue Speak)**: Gray, bottom strip

## How to Use

### Starting the Enhanced Demo

1. **Using the startup script:**
   ```bash
   ./start_enhanced_demo.sh
   ```

2. **Manual startup:**
   ```bash
   python bin/server.py \
       --model_path ./checkpoints \
       --llm_path ./Qwen2-7B-Instruct \
       --ip localhost \
       --port 8765 \
       --max_users 1
   ```

### Accessing the Interface

- **Basic Demo**: `https://localhost:8765/`
- **Enhanced Demo**: `https://localhost:8765/enhanced`

### Interaction Flow

1. **Set System Prompt** (optional): Enter a prompt in the text area and click "Set Prompt"
2. **Start Dialogue**: Click "Start Dialogue" to begin interaction
3. **Speak**: Talk into your microphone
4. **Observe States**: Watch the real-time state indicators and timeline graphs
5. **Stop Dialogue**: Click "Stop Dialogue" when done

## Understanding the Visualizations

### Normal Conversation Flow
1. User starts speaking → VAD activates → Dialog state: `dialog_sl` → `dialog_cl`
2. User stops speaking → VAD deactivates → Dialog state: `dialog_ss`
3. System generates response → Generation active → Dialog state: `dialog_cs`
4. System finishes → Generation inactive → Dialog state: `dialog_sl`

### Interruption Scenarios
- User can interrupt system generation (overlapping yellow and green bars)
- System intelligently distinguishes between backchannels and true interruptions
- Timeline shows the exact timing of state transitions

### State Prediction Intelligence
- The system uses an LLM-based state predictor to determine when to transition between states
- `dialog_cl` continues as long as the system believes the user is still speaking
- `dialog_ss` triggers when the system decides to start responding
- `dialog_el` occurs when the system detects end of speech but decides not to respond

## Technical Details

### Update Frequency
- State updates: Real-time (as events occur)
- Timeline updates: Every 100ms
- Visual refresh: 60fps for smooth animations

### Data Retention
- Timeline data: 10 seconds (100 data points at 100ms intervals)
- State history: Maintained in browser memory during session

### Browser Compatibility
- Requires modern browser with WebRTC support
- Tested on Chrome, Firefox, Safari
- HTTPS required for microphone access

## Troubleshooting

### Common Issues
1. **Microphone not working**: Ensure HTTPS and microphone permissions
2. **No state updates**: Check browser console for connection errors
3. **Poor audio quality**: Adjust browser audio settings

### Performance
- Timeline updates are optimized to avoid performance impact
- Chart animations use efficient update modes
- Memory usage is bounded by 10-second data retention

## Use Cases

### Research and Development
- Analyze system behavior during different interaction patterns
- Debug state machine transitions
- Study interruption handling mechanisms
- Visualize latency and timing characteristics

### Demonstration and Education
- Show real-time AI system operation
- Explain dialogue system architecture
- Demonstrate advanced speech interaction capabilities

### System Monitoring
- Monitor system health and responsiveness
- Identify timing issues or state machine problems
- Debug complex interaction scenarios
