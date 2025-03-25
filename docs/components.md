# Connectome Architecture: Component Classification and Message Flow

## Component Classification

### Fundamental (F)
Core architectural elements that rarely change; modifications represent major framework revisions
- Space and Element concepts
- Event propagation core
- Shell-Element relationship model
- Activity layer abstraction
- Timeline/Loom management (event DAG structure)

### Stable (S)
Important patterns that evolve with backward compatibility
- Rendering API
- Element Delegate interface
- Normalized event protocols
- Space relationship models (mounting, focus)
- Compression API

### Dynamic (D)
Swappable implementations that can vary without affecting architecture
- Specific Shell implementations (two-phase, single-phase)
- Element implementations
- Compression algorithms
- Specific adapters
- UI rendering strategies

> **Note**: The sequence diagrams and message flow descriptions below primarily illustrate the first of two possible multiuser chat models, where chat elements are mounted directly in the agent's Inner Space. For a comparison of both possible architectural approaches (direct chat elements vs. remote shared spaces with uplinks), please see the **[Multiuser Chat Participation Models](#multiuser-chat-participation-models)** section at the end of this document.

# Message Processing Flow (with Action Recording)

```mermaid
sequenceDiagram
    participant EA as External Adapter
    participant AL as Activity Layer
    participant Space
    participant Element
    participant ED as Element Delegate
    participant Shell
    participant HUD
    participant Agent
    
    %% External message reception
    EA->>AL: receiveExternalEvent(externalMsg)
    activate AL
    Note over AL: Normalizes to internal format
    Note over AL: Determines timeline context
    
    %% Space processing
    AL->>Space: receiveEvent(event, timelineContext)
    activate Space
    Note over Space: Validates timeline coherence
    Note over Space: Stores event in DAG
    Space->>Element: receiveEvent(eventData)
    activate Element
    
    %% Element processing
    Note over Element: Updates internal state
    Element-->>Space: notifyStateChanged()
    Space->>Space: storeStateForTimeline(elementId, state, timelineContext)
    deactivate Element
    
    %% Shell notification
    Space-->>Shell: notifyStateChange(notification, timelineContext)
    deactivate Space
    deactivate AL
    
    %% Context rendering
    activate Shell
    Note over Shell: Transitions to appropriate phase
    Shell->>HUD: prepareContextRendering(timelineContext)
    activate HUD
    
    %% Get rendering from elements
    loop For each relevant Space/Element
        HUD->>Space: renderElementInTimeline(elementId, options, timelineContext)
        activate Space
        Space->>ED: render(timelineState, options)
        activate ED
        Note over ED: Creates renderings from state
        ED-->>Space: renderingResult
        deactivate ED
        Space-->>HUD: elementRendering
        deactivate Space
    end
    
    %% Compression
    Note over HUD: Applies compression to fit context window
    HUD-->>Shell: finalContext
    deactivate HUD
    Shell->>Shell: applyCompression(finalContext, timelineContext)

    %% Agent processing
    Shell->>Agent: presentContext(compressedContext)
    activate Agent
    Note over Agent: Processes context (timeline-unaware)
    Agent-->>Shell: agentResponse
    deactivate Agent
    
    %% Action parsing
    Note over Shell: Parses agent actions
    
    %% Action execution
    alt Space action
        Shell->>Space: executeAction(action, timelineContext)
        activate Space
        Note over Space: Validates timeline coherence
        
        %% ADDED: Recording agent action in timeline
        rect rgb(240, 240, 255)
            Note over Space: Records agent action as event in timeline DAG
            Space->>Space: addEventToTimeline(agentActionEvent, timelineContext)
            Note over Space: Updates timeline context with new event
        end
        
        Note over Space: Updates element state for timeline
        Space-->>Shell: actionResult
        deactivate Space
    else Shell tool
        Shell->>Shell: executeShellTool(action, timelineContext)
        Note over Shell: Processes tool (e.g., bookmark, fork)
    end
    
    %% External propagation
    alt Primary timeline
        Shell->>Space: sendMessage(message, timelineContext)
        activate Space
        Note over Space: Stores event in DAG
        Space->>AL: sendMessage(message, timelineContext)
        deactivate Space
        activate AL
        AL->>EA: sendToExternal(externalMsg)
        deactivate AL
    else Non-primary timeline
        Note over Shell: Actions not propagated externally
    end
    
    %% Prepare for next cycle
    Note over Shell: Stores context for next cycle
    deactivate Shell
```

## Message Flow Sequence with Classification

### 1. External Event Reception (F/S/D)
```
// External adapter receives message (D)
Telegram_Adapter.receiveMessage(externalMessage)

// Normalization to standard protocol (S)
normalized = Adapter.normalizeToMessageProtocol(externalMessage)

// Determine timeline context (F)
timelineContext = {
  branchId: "primary-branch-001",
  isPrimary: true,
  lastEventId: "evt-45678"
}

// Routing through fundamental activity layer (F)
ActivityLayer.routeEvent(normalized, timelineContext)
```

The adapter implementation is Dynamic, but it conforms to a Stable protocol. The Activity Layer's existence and role is Fundamental.

### 2. Space Routing and Event Processing (F)
```
// Resolving target Space (F)
targetSpace = SpaceRegistry.resolveSpace(normalized.destination)

// Space receives event with timeline context (F)
targetSpace.receiveEvent(normalized.eventData, timelineContext)

// Validate timeline coherence (F)
if (targetSpace.isCoherentTimeline(timelineContext)) {
  // Space resolves timeline-specific element instance (F)
  chatElement = targetSpace.getElementForTimeline("main_chat", timelineContext)
  chatElement.processEvent(normalized.eventData)
}
```

The routing mechanism between Spaces and Elements is Fundamental to the architecture. How a specific Space resolves which Element should receive an event is also Fundamental.

### 3. Element State Update (S/D)
```
// Chat Element processes event (D)
ChatElement.processEvent(eventData) {
  // Implementation-specific state update
  this.addMessageToHistory(eventData)
  
  // Conforming to stable Element interface (S)
  this.notifyStateChanged()
}
```

The specific implementation of the Chat Element is Dynamic, but it adheres to a Stable interface pattern for all Elements. The concept that Elements manage their own state is Stable.

### 4. Shell Notification (F/S)
```
// Element notifies Shell of state change (F)
Element.notifyStateChanged()

// Space passes timeline context to Shell (F)
Space.notifyShellOfUpdate(elementId, timelineContext)

// Shell receives notification with timeline context (F)
Shell.handleElementStateChange(elementId, spaceId, timelineContext)

// Shell decides whether to update context (S)
if (Shell.shouldUpdateContext()) {
  Shell.initiateContextUpdate(timelineContext)
}
```

The notification pathway between Elements and the Shell is Fundamental. The decision logic for context updates falls into Stable territory - different Shell implementations can handle this differently.

### 5. Shell Phase Management (D)
```
// Two-Phase Shell implementation (D)
TwoPhaseShell.initiateContextUpdate() {
  // Implementation-specific phase transition
  this.currentPhase = "contemplation"
  
  // Calls to stable Shell APIs (S)
  this.renderCurrentContext()
}
```

The specific implementation of a two-phase interaction model is Dynamic. Different Shells could implement single-phase interaction, guided exploration, or other models while still using the Stable shell APIs.

### 6. Rendering Request Flow (F/S)
```
// Shell initiates rendering (F)
Shell.renderCurrentContext() {
  // Collect render targets (F)
  renderTargets = this.collectRenderTargets()
  
  // Request rendering from Elements (S)
  for (target of renderTargets) {
    elementRender = target.element.delegate.render(renderOptions)
    renderCollection.add(elementRender)
  }
}
```

The concept that the Shell controls rendering and collects from Elements is Fundamental. The specific Rendering API that delegates implement is Stable.

### 7. Element Delegate Rendering (S/D)
```
// Element delegate conforming to stable interface (S)
ChatDelegate.render(options) {
  // Implementation-specific rendering logic (D)
  renderedContent = this.formatMessages(this.element.getMessages())
  
  // Returns in standardized format (S)
  return {
    content: renderedContent,
    metadata: {
      type: "chat_history",
      importance: this.calculateImportance(),
      compressionHints: this.generateCompressionHints()
    }
  }
}
```

The Delegate interface is Stable, but specific rendering implementations are Dynamic. The metadata structure for rendering is also part of the Stable API.

### 8. Compression System (S/D)
```
// Shell applies compression using stable API (S)
Shell.applyCompression(renderCollection) {
  // Using Dynamic compression implementation (D)
  compressor = this.compressionStrategy
  
  // Analysis phase using stable protocols (S)
  analysisResult = compressor.analyzeContent(renderCollection)
  
  // Strategy determination (D)
  strategy = compressor.determineStrategy(analysisResult)
  
  // Execution of strategy through stable API (S)
  compressedContent = compressor.executeStrategy(renderCollection, strategy)
  
  return compressedContent
}
```

The Compression API and interfaces are Stable, while specific compression strategies and algorithms are Dynamic. The concept of compression itself is Stable, though specific compression algorithms and priorities can be Dynamic.

### 9. Context Assembly and Presentation (F/S/D)
```
// Final assembly (S)
Shell.assembleContext(compressedContent) {
  // Implementation-specific formatting (D)
  formattedContext = this.formatter.format(compressedContent)
  
  // Using fundamental agent interface (F)
  this.agent.presentContext(formattedContext)
}
```

The concept that the Shell assembles and presents context is Fundamental, but the specific formatting is Dynamic. The interface to the agent is part of the Fundamental architecture.

### 10. Agent Processing (F/D)
```
// Agent interface (F)
Agent.processContext(context) {
  // Implementation-specific processing (D)
  agentResponse = this.model.generate(context)
  
  // Return through fundamental interface (F)
  Shell.receiveAgentResponse(agentResponse)
}
```

The Agent interface is Fundamental, but specific model implementations are Dynamic. The Shell's ability to receive responses from the Agent is Fundamental.

### 11. Action Parsing and Execution (S/D)
```
// Shell processes agent response (S)
Shell.receiveAgentResponse(response) {
  // Dynamic implementation of response parsing (D)
  actions = this.actionParser.parse(response)
  
  // Execution through stable interface (S)
  for (action of actions) {
    if (action.isElementAction()) {
      this.executeElementAction(action)
    } else if (action.isShellAction()) {
      this.executeShellAction(action)
    }
  }
}
```

The action parsing interface is Stable, but specific parsing implementations are Dynamic. The distinction between Element actions and Shell actions is part of the Stable architecture.

### 12. Element Action Execution (F/S)
```
// Shell routes action to Element (F)
Shell.executeElementAction(action) {
  // Resolve target Element (F)
  targetElement = this.getSpace(action.spaceId).getElement(action.elementId)
  
  // Execute using Element interface (S)
  result = targetElement.executeAction(action.name, action.parameters)
  
  // Handle result (S)
  if (result.requiresUpdate) {
    this.scheduleContextUpdate()
  }
}
```

The routing of actions to Elements is Fundamental, while the specific action interface is Stable. The ability of actions to trigger context updates is also part of the Stable layer.

### 13. Message Creation and Propagation (F/S/D)
```
// Chat Element creates message (D)
ChatElement.executeAction("send_message", params) {
  // Implementation-specific message creation (D)
  newMessage = this.createMessage(params.content)
  this.addMessageToHistory(newMessage)
  
  // Stable notification protocol (S)
  this.notifyStateChanged()
  
  // Fundamental event propagation with timeline context (F)
  this.propagateExternalEvent(newMessage, timelineContext)
}
```

The specific implementation of the Chat Element is Dynamic, but it adheres to Stable notification protocols. The concept of event propagation is Fundamental.

### 14. Activity Layer Propagation (F/S)
```
// Element initiates external propagation (F)
Element.propagateExternalEvent(event, timelineContext) {
  // Through Space (F)
  this.space.propagateExternalEvent(this.id, event, timelineContext)
}

// Space routes to Activity Layer (F)
Space.propagateExternalEvent(elementId, event, timelineContext) {
  ActivityLayer.propagateEvent(this.id, elementId, event, timelineContext)
}

// Activity Layer normalizes and routes (S/F)
ActivityLayer.propagateEvent(spaceId, elementId, event, timelineContext) {
  // Using stable normalization protocol (S)
  normalized = this.normalizeForExternal(event, spaceId, elementId)
  
  // Check if this is primary timeline (F)
  if (timelineContext.isPrimary) {
    // Fundamental routing to adapters (F)
    this.routeToAdapters(normalized)
  } else {
    // Log but don't externalize non-primary timeline events
    logger.info(`Event in timeline ${timelineContext.branchId} not propagated (non-primary)`)
  }
}
```

The propagation pathway through Elements, Spaces, and the Activity Layer is Fundamental. The normalization protocol is Stable, allowing for evolution without breaking the architecture.

### 15. Adapter Processing and External Delivery (S/D)
```
// Activity Layer routes to appropriate adapters (F)
ActivityLayer.routeToAdapters(normalized) {
  // Find adapters for destination (S)
  targetAdapters = AdapterRegistry.findForDestination(normalized.destination)
  
  // Each adapter handles format conversion (D)
  for (adapter of targetAdapters) {
    externalFormat = adapter.formatForExternal(normalized)
    adapter.sendToExternal(externalFormat)
  }
}
```

The routing concept is Fundamental, while the adapter selection mechanism is Stable. The specific formatting and sending implementations are Dynamic.

### 16. Cycle Completion and State Update (F/S)
```
// Shell updates state for next cycle with timeline context (F)
Shell.completeActionCycle(timelineContext) {
  // Update fundamental state (F)
  this.lastActionTime = Date.now()
  this.currentTimelineContext = timelineContext
  
  // Implementation-specific next actions (D)
  if (this.hasPendingUpdates()) {
    this.processNextUpdate()
  } else {
    this.waitForNextEvent()
  }
}
```

The concept of cycle completion is Fundamental, while the specific next steps vary by Shell implementation (Dynamic).

## Key Architecture Insights

1. **Layered Stability**
   - Fundamental components form the backbone, rarely changing
   - Stable components allow for evolution without breaking changes
   - Dynamic components encourage diverse implementations and experimentation

2. **Protocol-Based Interoperability**
   - Standard protocols enable component substitution
   - Elements can be swapped while preserving interface contracts
   - Different Shell implementations can interoperate with the same Spaces

3. **Separation of Rendering and State**
   - Elements own their state (Fundamental)
   - Delegates control rendering (Stable interface, Dynamic implementation)
   - Shell manages assembly and compression (Stable API, Dynamic algorithms)

4. **Space-Element-Shell Relationship**
   - Spaces provide organizational structure (Fundamental)
   - Elements provide functionality (Dynamic implementation, Stable interface)
   - Shell provides runtime environment and coordination (Fundamental concept, Dynamic implementation)

5. **Extensibility Patterns**
   - New normalized protocols can be added (Stable)
   - New adapter types can be created (Dynamic)
   - New Shell implementations can be developed (Dynamic)
   - New Space types can be designed (Dynamic implementation, Fundamental concept)

This architecture provides a clear separation between the foundational concepts that define the system, the stable interfaces that enable component interoperability, and the dynamic implementations that allow for innovation and customization.

# Multiuser Chat Participation Models

The framework supports two distinct architectural approaches for agent participation in multiuser chats, each with different implications for history management, context rendering, and system scalability.

## Model 1: Direct Chat Elements in Inner Space

In the first model, which is the primary focus of the sequence diagrams shown earlier, chat elements are mounted directly in the agent's Inner Space. External adapters normalize events from various platforms and route them to the appropriate chat elements.

```mermaid
graph TD
    subgraph "Agent's Inner Space"
        IM[Inner Space Timeline]
        S[Shell]
        CM[Context Manager]
        C1[Telegram Chat Element]
        C2[Discord Chat Element]
        C3[Slack Chat Element]
    end
    
    subgraph "Activity Layer"
        AL[Activity Layer]
        TA[Telegram Adapter]
        DA[Discord Adapter]
        SA[Slack Adapter]
    end
    
    subgraph "External Systems"
        TE[Telegram]
        DE[Discord]
        SE[Slack]
    end
    
    TE --> TA
    DE --> DA
    SE --> SA
    
    TA --> AL
    DA --> AL
    SA --> AL
    
    AL --> C1
    AL --> C2
    AL --> C3
    
    C1 --> S
    C2 --> S
    C3 --> S
    
    S --> IM
```

### Key Characteristics:

1. **Direct Integration**: Chat elements are mounted directly in the agent's Inner Space
2. **Timeline Ownership**: All events become part of the agent's subjective timeline in their Inner Space
3. **Adapter-Centric**: Each external system requires a specific adapter to normalize events
4. **Singular History**: Chat history is part of the agent's personal history DAG
5. **Compression Handling**: Chat history is compressed using standard subjective timeline compression techniques
6. **Implementation Complexity**: Each agent needs adapters for every platform they interact with

## Model 2: Uplinks to Remote Shared Spaces

In the second model, agents connect to remote shared spaces through uplinks. These shared spaces maintain their own timelines and can service multiple agents simultaneously.

```mermaid
graph TD
    subgraph "Agent 1's Inner Space"
        IM1[Inner Space Timeline]
        S1[Shell]
        UP1[UplinkProxy]
    end
    
    subgraph "Agent 2's Inner Space"
        IM2[Inner Space Timeline]
        S2[Shell]
        UP2[UplinkProxy]
    end
    
    subgraph "Shared Space: Telegram Community"
        SST[Shared Space Timeline]
        TC[Telegram Chat Element]
        CS1[Connection Span - Agent 1]
        CS2[Connection Span - Agent 2]
    end
    
    subgraph "Activity Layer"
        AL[Activity Layer]
        TA[Telegram Adapter]
    end
    
    subgraph "External Systems"
        TE[Telegram]
    end
    
    TE --> TA
    TA --> AL
    AL --> TC
    
    TC --> SST
    
    UP1 --> SST
    UP2 --> SST
    
    CS1 --> UP1
    CS2 --> UP2
    
    UP1 --> S1
    UP2 --> S2
    
    S1 --> IM1
    S2 --> IM2
```

### Key Characteristics:

1. **Space Separation**: Chat elements exist in separate shared spaces, not in the agent's Inner Space
2. **Connection Spans**: The agent's Inner Space only records connection events (when connected/disconnected)
3. **History Bundles**: History from remote spaces is retrieved as bundles during context rendering
4. **Specialized Compression**: Remote history bundles can be partially compressed:
   - Recent spans may be preserved in full detail
   - Mid-term spans may keep only question-answer pairs
   - Historical spans may be heavily summarized
5. **Efficient Multiagent Support**: Multiple agents can uplink to the same shared space
6. **Reduced Adapter Complexity**: Agents don't need adapters for every platform, just uplink capability
7. **Thread-Aware Processing**: Conversational threads can be preserved during compression

## Model Comparison: Message Processing Flow

The following diagrams illustrate the difference in message flow between the two models:

### Model 1: Direct Chat Processing
```mermaid
sequenceDiagram
    participant E as External System
    participant A as Adapter
    participant AL as Activity Layer
    participant CE as Chat Element (Inner Space)
    participant S as Shell
    participant Ag as Agent
    
    E->>A: External Message
    A->>AL: Normalized Message
    AL->>CE: Internal Event
    Note over CE: Updates state directly in agent's timeline
    CE->>S: Notify state change
    S->>Ag: Update context (full history in timeline)
    Ag->>S: Response
    S->>CE: Execute action
    CE->>AL: Propagate event
    AL->>A: Format for external
    A->>E: External message
```

### Model 2: Remote Space with Uplink
```mermaid
sequenceDiagram
    participant E as External System
    participant A as Adapter
    participant AL as Activity Layer
    participant RS as Remote Shared Space
    participant CE as Chat Element (Remote)
    participant UP as UplinkProxy
    participant S as Shell
    participant Ag as Agent
    
    E->>A: External Message
    A->>AL: Normalized Message
    AL->>RS: Internal Event
    RS->>CE: Update chat element
    
    Note over RS: Stores event in remote space timeline
    
    RS->>UP: Track event in connection span
    Note over UP: Doesn't copy event, just tracks span reference
    
    UP->>S: Notify state change
    
    Note over S: During context rendering...
    S->>UP: Get connection spans
    UP->>RS: Request history for spans
    RS->>UP: Return history bundles
    
    Note over UP: History bundles can be partially compressed
    
    UP->>S: Return processed bundles
    S->>Ag: Update context (clear boundary between subjective/remote)
    
    Ag->>S: Response
    S->>RS: Execute action in remote space
    RS->>CE: Update chat element
    RS->>AL: Propagate event
    AL->>A: Format for external
    A->>E: External message
```

## Implications of the Two Models

1. **Memory Efficiency**:
   - Model 1 stores all history in the agent's timeline, which can become large
   - Model 2 only stores connection references, retrieving history on demand

2. **Context Clarity**:
   - Model 1 blends all chat history into the agent's subjective experience
   - Model 2 maintains clear boundaries between the agent's subjective timeline and remote observations

3. **History Compression**:
   - Model 1 applies standard compression to all history
   - Model 2 enables specialized partial compression of remote history bundles

4. **Scalability**:
   - Model 1 requires each agent to implement adapters for every system
   - Model 2 allows many agents to share a single implementation of external adapters

5. **Multi-agent Collaboration**:
   - Model 1 has no built-in support for agent-to-agent interaction
   - Model 2 naturally enables multiple agents to interact in shared spaces

Both models have valid use cases within the framework, and implementations can choose the appropriate model based on their specific requirements for memory efficiency, context management, and multi-agent collaboration.