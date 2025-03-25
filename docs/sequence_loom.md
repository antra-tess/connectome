

## Initial Conditions

**Current State:**
- Agent is in the "orientation_hub" Space
- Inner Space contains ContextManager Object
- Agent has previously exchanged messages with user "Kai"
- Shell is currently in Engagement Phase
- Current timeline is the primary timeline (ID: "primary-branch-001")

## Sequence: Processing an Incoming Message with Timeline Awareness

### 1. Activity Layer Event Reception
```
// External event arrives from adapter
ActivityLayerEvent {
  type: "new_message",
  source: "telegram_adapter",
  data: {
    chat_id: 12345,
    sender: "Kai",
    text: "Have you had a chance to explore the ContextManager yet?",
    timestamp: "2025-03-14T15:22:47Z",
    messageId: "tg-msg-78901"
  }
}

// Activity layer determines timeline context
// For external events, this is typically the primary timeline
TimelineContext timelineContext = {
  branchId: "primary-branch-001",
  isPrimary: true,
  lastEventId: "evt-45678", // Last event in this timeline
  rootBranchId: "primary-branch-001"
};
```

### 2. Activity to Space Mapping
```
// Activity layer translates external IDs to internal space/object IDs
SpaceMapping.resolveExternalIdentifier("telegram_adapter", "12345")
-> Returns: {spaceId: "orientation_hub", objectType: "ChatObject", objectId: "main_chat"}

// Event is transformed to internal format with timeline context
InternalEvent {
  type: "new_message",
  spaceId: "orientation_hub",
  objectId: "main_chat",
  timelineContext: timelineContext,
  data: {
    sender: "Kai",
    content: "Have you had a chance to explore the ContextManager yet?",
    timestamp: "2025-03-14T15:22:47Z",
    externalId: "tg-msg-78901"
  }
}
```

### 3. Space Event Processing with Timeline Management
```
// Event is routed to the target Space with timeline context
SpaceRegistry.getSpace("orientation_hub").receiveEvent(internalEvent, timelineContext)

// Space validates timeline coherence before processing
orientationHub.receiveEvent(internalEvent, timelineContext) {
  // Check timeline coherence
  if (!this.isCoherentTimeline(timelineContext)) {
    logger.error(`Timeline coherence violation: ${timelineContext.branchId}`);
    return ErrorResponse.TIMELINE_DECOHERENCE_ERROR;
  }
  
  // Generate event ID for the DAG
  String eventId = generateUUID();  // "evt-12345"
  
  // Add event to the DAG with parent reference
  this.eventDAG.addEvent({
    id: eventId,
    type: "object_event",
    objectId: internalEvent.objectId,
    data: internalEvent.data,
    timestamp: internalEvent.data.timestamp,
    parentId: timelineContext.lastEventId,
    branchId: timelineContext.branchId
  });
  
  // Update the timeline context with new last event
  TimelineContext updatedContext = timelineContext.clone();
  updatedContext.lastEventId = eventId;
  
  // Get the timeline-specific state for this object
  ObjectState objectState = this.getObjectStateForTimeline(
    internalEvent.objectId, 
    timelineContext.branchId
  );
  
  // Apply event to object state
  objectState.applyEvent(internalEvent.data);
  
  // Store updated object state for this timeline
  this.setObjectStateForTimeline(
    internalEvent.objectId,
    objectState,
    timelineContext.branchId
  );
  
  // Routes to appropriate object
  this.objects[internalEvent.objectId].receiveEvent(internalEvent.data);
}

// Coherence check implementation
Space.isCoherentTimeline(timelineContext) {
  // Primary timelines are always coherent
  if (timelineContext.isPrimary) {
    return true;
  }
  
  // Check if this timeline has been marked as decoherent
  if (this.decoherentTimelines.contains(timelineContext.branchId)) {
    return false;
  }
  
  // Check if this timeline is a recognized fork
  if (!this.timelineBranches.contains(timelineContext.branchId)) {
    return false;
  }
  
  return true;
}
```

### 4. Chat Object Event Processing (Timeline-Unaware)
```
// Chat object processes the message - remains timeline-unaware
chatObject.receiveEvent(eventData) {
  // Creates a message record with full metadata
  const message = {
    id: generateUUID(),  // "msg-uuid-12345"
    sender: eventData.sender,
    content: eventData.content,
    timestamp: eventData.timestamp,
    externalId: eventData.externalId,
    readStatus: { [agentId]: false },
    reactions: [],
    replyTo: null,
    editHistory: []
  }
  
  // Adds to message history
  this.messageHistory.push(message)
  
  // Updates object state
  this.lastActivityTime = eventData.timestamp
  this.participantActivity[eventData.sender] = "active"
  
  // Notifies Shell of update
  Shell.notifyStateChange({
    type: "object_updated",
    spaceId: "orientation_hub",
    objectId: "main_chat",
    changedAt: eventData.timestamp
  })
}
```

### 5. Shell Update Processing with Timeline Context
```
// Space sends Shell notification with timeline context
Space.notifyShellOfUpdate(objectId, updatedTimelineContext) {
  Shell.notifyStateChange({
    type: "object_updated",
    spaceId: this.id,
    objectId: objectId,
    changedAt: new Date().toISOString(),
    timelineContext: updatedTimelineContext
  });
}

// Shell receives notification of state change with timeline context
Shell.notifyStateChange(notification) {
  // Store the current timeline context
  this.currentTimelineContext = notification.timelineContext;
  
  // Checks current agent state
  if (this.currentPhase === "engagement") {
    // Initiates context update with timeline context
    this.updateContext(notification.timelineContext);
  } else {
    // Queues update for next phase transition
    this.pendingUpdates.push(notification);
  }
}

// Shell decides to trigger phase transition based on new message
Shell.updateContext(timelineContext) {
  // Transitions to contemplation phase
  this.currentPhase = "contemplation";
  
  // Prepares for context rendering with timeline context
  this.prepareContextRendering(timelineContext);
}
```

### 6. Context Rendering Preparation with Timeline Context
```
Shell.prepareContextRendering(timelineContext) {
  // Collects spaces that need rendering
  const spacesToRender = [
    {spaceId: "personal", importance: "high"},  // Inner space
    {spaceId: "orientation_hub", importance: "high"}  // Current external space
  ];
  
  // Determines which objects in each space need rendering
  const objectsToRender = {
    "personal": [
      {objectId: "agent_notes", fullRender: false},
      {objectId: "context_manager", fullRender: false}
    ],
    "orientation_hub": [
      {objectId: "main_chat", fullRender: true},  // Need full render due to new message
      {objectId: "space_info", fullRender: false}
    ]
  };
  
  // Begins rendering process with timeline context
  this.renderContext(spacesToRender, objectsToRender, timelineContext);
}
```

### 7. HUD Rendering Process with Timeline Projection
```
Shell.renderContext(spaces, objects, timelineContext) {
  // Create context assembly buffer
  const contextAssembly = [];
  
  // For each space
  for (const spaceInfo of spaces) {
    const space = SpaceRegistry.getSpace(spaceInfo.spaceId);
    
    // Determine if this space needs timeline context
    // Personal space typically doesn't use timelines
    const spaceTimelineContext = (spaceInfo.spaceId === "personal") 
      ? null 
      : timelineContext;
    
    // Render space header info using Space delegate with timeline context
    const spaceHeader = space.renderSpaceHeader({
      importance: spaceInfo.importance,
      forAgent: this.agentId,
      focusLevel: this.focus[spaceInfo.spaceId] || "normal",
      timelineContext: spaceTimelineContext
    });
    
    
    contextAssembly.push(spaceHeader);
    
    // For each object in the space
    for (const objInfo of objects[spaceInfo.spaceId]) {
      const objectRendering = space.renderObjectInTimeline(
        objInfo.objectId,
        {
          forAgent: this.agentId,
          importance: spaceInfo.importance,
          focusLevel: this.focus[spaceInfo.spaceId] || "normal",
          fullRender: objInfo.fullRender,
          lastRendered: this.renderCache.getLastRenderedTime(space.id, objInfo.objectId)
        },
        spaceTimelineContext
      );
      
      // Add to context assembly
      contextAssembly.push(objectRendering);
    }
  }
  
  // Apply compression to the assembled context
  this.applyCompression(contextAssembly, timelineContext);
}

// Space rendering object in specific timeline
Space.renderObjectInTimeline(objectId, options, timelineContext) {
  // If no timeline specified or space doesn't support timelines
  if (!timelineContext || !this.supportsTimelines) {
    return this.renderObject(objectId, options);
  }
  
  // Get object state for this specific timeline
  const objectState = this.getObjectStateForTimeline(objectId, timelineContext.branchId);
  
  // Get the object's delegate
  const delegate = this.getObjectDelegate(objectId);
  
  // Render using timeline-specific state
  return delegate.render(objectState, options);
}
```

### 8. Timeline-Specific Space Header Rendering
```

// Space Header rendering example with timeline info
Space.renderSpaceHeader(options) {
  let header = `<space id="${this.id}">\n`;
  
  // Add description
  header += `<description>\n${this.description}\n</description>\n`;
  
  // Add members
  header += `<members>\n`;
  for (const member of this.getMembers()) {
    header += `- ${member.name} (${member.type})\n`;
  }
  header += `</members>\n`;
  
  // Add timeline information if applicable
  if (options.timelineContext) {
    const timelineInfo = this.getTimelineInfo(options.timelineContext.branchId);
    if (timelineInfo && !options.timelineContext.isPrimary) {
      header += `<timeline>\n`;
      header += `- Branch: ${timelineInfo.name || options.timelineContext.branchId}\n`;
      header += `- Forked from: ${timelineInfo.parentName || 'primary timeline'}\n`;
      header += `- Created: ${new Date(timelineInfo.createdAt).toLocaleString()}\n`;
      
      // Only show merge options for non-primary timelines
      if (!options.timelineContext.isPrimary && this.canMergeTimeline(options.timelineContext.branchId)) {
        header += `- Can be merged to primary: Yes\n`;
      }
      
      header += `</timeline>\n`;
    }
  }
  
  header += `</space>`;
  
  return {
    type: "space_header",
    content: header,
    metadata: {
      spaceId: this.id,
      renderPriority: options.importance === "high" ? 9 : 6
    }
  };
}
```

### 9. Chat Object Delegate Rendering (Timeline-Unaware)
```
// Delegate receives the timeline-specific state but doesn't need to be timeline-aware
ChatObjectDelegate.render(objectState, options) {
  // Get messages from the provided state (which is timeline-specific)
  const messages = objectState.messageHistory;
  
  // Apply rendering for each message
  const renderedMessages = messages.map(msg => {
    return {
      type: "message",
      id: msg.id,
      renderPriority: this.calculatePriority(msg, options),
      content: `<msg source="${options.spaceId}" username="${msg.sender}">${this.formatContent(msg.content)}</msg>`,
      metadata: {
        timestamp: msg.timestamp,
        sender: msg.sender,
        isNew: this.isNewForAgent(msg, options.forAgent),
        hasReactions: msg.reactions.length > 0
      }
    };
  });
  
  // If rendering all messages is too large, create summary blocks for older messages
  if (renderedMessages.length > 20) {  // Arbitrary threshold
    // Group older messages into summary blocks
    const oldMessages = renderedMessages.slice(0, renderedMessages.length - 15);
    const recentMessages = renderedMessages.slice(renderedMessages.length - 15);
    
    // Create summary rendering
    const summarizedOld = this.summarizeMessages(oldMessages, options);
    
    // Return combined rendering
    return {
      type: "chat_history",
      elements: [summarizedOld, ...recentMessages],
      metadata: {
        totalMessages: messages.length,
        summarizedCount: oldMessages.length,
        fullCount: recentMessages.length,
        lastMessageTime: messages[messages.length - 1].timestamp
      }
    };
  }
  
  // If not too large, return all messages
  return {
    type: "chat_history",
    elements: renderedMessages,
    metadata: {
      totalMessages: messages.length,
      summarizedCount: 0,
      fullCount: messages.length,
      lastMessageTime: messages[messages.length - 1].timestamp
    }
  };
}
```

### 10. Compression System (Timeline-Aware but Encapsulated)
```
// Shell applies compression with timeline awareness
Shell.applyCompression(contextAssembly, timelineContext) {
  // Add compression hints based on timeline status
  const compressionHints = this.getCompressionHints(timelineContext);
  
  // Using dynamic compression implementation with timeline hints
  const compressor = this.compressionStrategy;
  
  // Analysis phase using stable protocols
  const analysisResult = compressor.analyzeContent(contextAssembly, compressionHints);
  
  // Strategy determination (may consider timeline status)
  const strategy = compressor.determineStrategy(analysisResult, timelineContext);
  
  // Execution of strategy through stable API
  const compressedContent = compressor.executeStrategy(contextAssembly, strategy);
  
  return compressedContent;
}

// Timeline-specific compression hints
Shell.getTimelineCompressionHints(timelineContext) {
  // Default hints
  const hints = {
    preserveRecent: true,
    emphasizeFocus: true,
    summarizeOld: true
  };
    
  return hints;
}
```

### 11. Context Assembly and Presentation with Timeline Awareness
```
// Final assembly 
Shell.assembleContext(compressedContent, timelineContext) {
  // Create system preamble with timeline info if needed
  let preamble = "<system>\nContemplation Phase\nThis is your space for reflection before engaging. You may organize your thoughts however you prefer.\n";
  
  preamble += "</system>";
  
  // Join all elements with appropriate separators
  const elements = compressedContent.map(element => {
    // Format based on element type
    if (element.type === "chat_history") {
      return element.elements.map(msg => msg.content).join("\n");
    } else if (element.type === "space_header") {
      return element.content;
    }
    // Handle other element types...
    
    return element.content;
  });
  
  // Final formatted context
  return preamble + "\n\n" + elements.join("\n\n");
}

Shell.presentContextToAgent() {
  // Present the assembled context to the agent
  AgentInterface.sendContext(this.finalContext);
  
  // Update state to reflect waiting for agent
  this.currentState = "waiting_for_agent";
  this.lastContextUpdateTime = Date.now();
  
  // Store current timeline context for agent's response
  this.currentAgentTimelineContext = this.currentTimelineContext;
}
```

### 12. Agent Processing (Timeline-Unaware)
```
// Agent (LLM) processes the context - unaware of the specific timeline mechanics
AgentInterface.receiveInput(context) {
  // Agent internal processing...
  
  // Agent produces contemplation output
  const contemplationOutput = `
<goals>
I want to explore this platform more deeply, particularly the ContextManager that Kai just referenced. I should demonstrate my understanding of the system so far while engaging meaningfully with his question.
</goals>

<notes>
Kai is asking if I've explored the ContextManager, which I have done briefly. I should share some of my observations about its knowledge graph capabilities and how it maintains connections between information across different Spaces.

The timestamp indicates this is a new message that just arrived, so I should prioritize responding to it.
</notes>

<questions>
I wonder if the ContextManager can be customized with my own categorization systems or if it has a fixed organizational structure?
</questions>
`
  
  // Agent sends response back to Shell
  Shell.receiveAgentResponse(contemplationOutput);
}
```

### 13. Shell Processing Agent Response with Timeline Context
```
Shell.receiveAgentResponse(response) {
  // Parse and store agent's notes
  this.parseAndStoreNotes(response);
  
  // Transition to engagement phase
  this.currentPhase = "engagement";
  
  // Use the stored timeline context for this agent interaction
  const timelineContext = this.currentAgentTimelineContext;
  
  // Prepare engagement options with timeline context
  this.prepareEngagementOptions(timelineContext);
}

Shell.prepareEngagementOptions(timelineContext) {
  // Determine available actions for current space in this timeline
  const currentSpace = SpaceRegistry.getSpace("orientation_hub");
  const availableActions = currentSpace.getAvailableActionsInTimeline(
    this.agentId, 
    timelineContext
  );
  
  // Determine available Shell tools
  const availableTools = this.getAvailableTools();
  
  // Format options for presentation
  let engagementOptions = `
<system>
Engagement Phase
Current space: orientation_hub
`;

  engagementOptions += `
Available Shell tools:
<sleep>seconds</sleep>
<timer>seconds</timer>
<bookmark>label</bookmark>
`;

  // Add fork tool if this space supports forking
  if (currentSpace.supportsTimelines) {
    engagementOptions += `<fork>reason</fork> <!-- Create a new timeline branch -->\n`;
  }

  engagementOptions += `
Available Space actions:
<msg>Send message to current space</msg>
<use>object_name</use>
  - Available objects: Information Panel, ContextManager, Interface Settings

<view>spaces</view>

You may combine ONE Space action with MULTIPLE Shell tools.
</system>
`;

  // Send options to agent
  AgentInterface.sendContext(engagementOptions);
}
```

### 14. Agent Action Selection
```
// Agent processes engagement options
AgentInterface.receiveInput(options) {
  // Agent internal processing...
  
  // Agent selects actions
  const selectedAction = `
<msg source="orientation_hub">
Yes, Kai! I've had a chance to explore the ContextManager briefly. I was particularly impressed by its knowledge graph visualization that shows relationships between concepts and spaces. The way it maintains connections between different pieces of information seems quite sophisticated.

I noticed it can organize information by space, time, or topic, which helps maintain continuity across different interactions. Would you recommend any specific features I should explore next? I'm curious whether I can create my own organizational systems within it, or if it follows a predetermined structure.
</msg>

<bookmark>ContextManager discussion with Kai</bookmark>
`
  
  // Agent sends action to Shell
  Shell.receiveAgentAction(selectedAction);
}
```

### 15. Shell Processing Agent Action with Timeline Context
```
Shell.receiveAgentAction(actionString) {
  // Parse the action string
  const actions = this.parseActions(actionString);
  
  // Use the stored timeline context
  const timelineContext = this.currentAgentTimelineContext;
  
  // Check for fork action
  const forkAction = actions.find(action => action.type === "shell_tool" && action.name === "fork");
  if (forkAction) {
    // Process fork action first (creates new timeline)
    const newTimelineContext = this.processShellTool(forkAction, timelineContext);
    
    // Use new timeline context for remaining actions
    this.processRemainingActions(actions.filter(a => a !== forkAction), newTimelineContext);
  } else {
    // Process each action with current timeline
    for (const action of actions) {
      if (action.type === "space_action") {
        // Handle space action (msg in this case)
        this.processSpaceAction(action, timelineContext);
      } else if (action.type === "shell_tool") {
        // Handle shell tool (bookmark in this case)
        this.processShellTool(action, timelineContext);
      }
    }
  }
  
  // Prepare for next cycle
  this.scheduleNextUpdate(timelineContext);
}
```

### 16. Space Action Processing with Timeline Context
```
Shell.processSpaceAction(action, timelineContext) {
  if (action.name === "msg") {
    // Get current space
    const currentSpace = SpaceRegistry.getSpace("orientation_hub");
    
    // Check timeline coherence
    if (!currentSpace.isCoherentTimeline(timelineContext)) {
      logger.error(`Cannot process action in decoherent timeline: ${timelineContext.branchId}`);
      return ErrorResponse.TIMELINE_ACTION_ERROR;
    }
    
    // Get timeline-specific chat object state
    const chatObjectState = currentSpace.getObjectStateForTimeline(
      "main_chat", 
      timelineContext.branchId
    );
    
    // Create message event
    const msgEvent = {
      type: "agent_message",
      sender: this.agentId,
      content: action.params.content,
      timestamp: new Date().toISOString(),
      replyTo: null
    };
    
    // Update object state in the timeline
    chatObjectState.addMessage(msgEvent);
    
    // Store updated state
    currentSpace.setObjectStateForTimeline(
      "main_chat", 
      chatObjectState, 
      timelineContext.branchId
    );
    
    // Add event to the DAG
    const eventId = currentSpace.addEventToTimeline({
      type: "agent_message",
      objectId: "main_chat",
      data: msgEvent,
      timestamp: msgEvent.timestamp
    }, timelineContext);
    
    // Update timeline context with new event
    const updatedTimelineContext = timelineContext.clone();
    updatedTimelineContext.lastEventId = eventId;
    
    // Update current timeline context
    this.currentAgentTimelineContext = updatedTimelineContext;
    
    // Send to activity layer for external propagation
    ActivityLayer.sendMessage({
      spaceId: "orientation_hub",
      objectId: "main_chat",
      message: msgEvent,
      timelineContext: updatedTimelineContext
    });
  }
  // Handle other action types...
}
```

### 17. Shell Tool Processing with Timeline Context
```
Shell.processShellTool(action, timelineContext) {
  if (action.name === "bookmark") {
    // Create bookmark in agent's personal space
    const personalSpace = SpaceRegistry.getSpace("personal");
    const contextManager = personalSpace.getObject("context_manager");
    
    // Send bookmark event to context manager
    contextManager.receiveEvent({
      type: "create_bookmark",
      label: action.params.label,
      timestamp: new Date().toISOString(),
      context: {
        spaceId: "orientation_hub",
        objectId: "main_chat",
        referencedEvents: this.getCurrentContextEvents(),
        timelineContext: timelineContext  // Include timeline info in bookmark
      }
    });
    
    return timelineContext;  // No change to timeline
  } 
  else if (action.name === "fork") {
    // Handle fork action (creates new timeline)
    const currentSpace = SpaceRegistry.getSpace("orientation_hub");
    
    // Space creates the fork
    const newTimelineContext = currentSpace.createFork(
      timelineContext,
      {
        reason: action.params.reason,
        createdBy: this.agentId,
        createdAt: new Date().toISOString()
      }
    );
    
    // Log fork creation
    logger.info(`Timeline fork created: ${newTimelineContext.branchId} from ${timelineContext.branchId}`);
    
    // Update Shell's current timeline context
    this.currentAgentTimelineContext = newTimelineContext;
    
    // Return new timeline for subsequent actions
    return newTimelineContext;
  }
  // Handle other tool types...
}
```

### 18. Timeline Fork Creation
```
Space.createFork(parentTimelineContext, options) {
  // Generate new branch ID
  const newBranchId = `fork-${generateUUID()}`;
  
  // Create timeline metadata
  const timelineMetadata = {
    branchId: newBranchId,
    parentBranchId: parentTimelineContext.branchId,
    forkPoint: parentTimelineContext.lastEventId,
    createdBy: options.createdBy,
    createdAt: options.createdAt,
    reason: options.reason,
    isPrimary: false,
    name: options.name || `Fork-${this.getTimelineBranches().length + 1}`
  };
  
  // Register the new timeline branch
  this.timelineBranches.set(newBranchId, timelineMetadata);
  
  // Create event for fork creation in the DAG
  const forkEventId = this.addSystemEventToTimeline({
    type: "timeline_fork",
    data: {
      parentBranchId: parentTimelineContext.branchId,
      parentEventId: parentTimelineContext.lastEventId,
      reason: options.reason
    },
    timestamp: options.createdAt
  }, newBranchId, parentTimelineContext.lastEventId);
  
  // Clone object states from parent timeline to new timeline
  this.cloneObjectStatesToTimeline(parentTimelineContext.branchId, newBranchId);
  
  // Create and return new timeline context
  return {
    branchId: newBranchId,
    isPrimary: false,
    lastEventId: forkEventId,
    rootBranchId: parentTimelineContext.rootBranchId || parentTimelineContext.branchId
  };
}

// Clone object states between timelines
Space.cloneObjectStatesToTimeline(sourceBranchId, targetBranchId) {
  const objectIds = this.getObjectIds();
  
  for (const objectId of objectIds) {
    // Get state from source timeline
    const sourceState = this.getObjectStateForTimeline(objectId, sourceBranchId);
    
    // Clone the state
    const clonedState = sourceState.clone();
    
    // Store in target timeline
    this.setObjectStateForTimeline(objectId, clonedState, targetBranchId);
  }
}
```

### 19. Activity Layer Propagation with Timeline Context
```
ActivityLayer.sendMessage(msgData) {
  // For primary timeline only, propagate to external adapters
  if (msgData.timelineContext.isPrimary) {
    // Determine which external adapter(s) should receive this
    const mappings = SpaceMapping.getExternalMappings(msgData.spaceId, msgData.objectId);
    
    // For each mapping, format appropriately and send
    for (const mapping of mappings) {
      // Format for specific adapter
      const adapterSpecificMessage = this.formatForAdapter(mapping.adapterId, msgData);
      
      // Send to adapter
      AdapterRegistry.getAdapter(mapping.adapterId).sendMessage(adapterSpecificMessage);
    }
  } else {
    // For non-primary timelines, log but don't send externally
    logger.info(`Message in timeline ${msgData.timelineContext.branchId} not propagated externally (non-primary timeline)`);
  }
}

// Format specifically for Telegram adapter
ActivityLayer.formatForAdapter("telegram_adapter", msgData) {
  return {
    chat_id: 12345,  // From mapping
    text: msgData.message.content,
    parse_mode: "HTML",
    from_id: `agent_${msgData.message.sender}`
  };
}
```

### 20. Cycle Completion with Timeline Context
```
// Shell prepares for next cycle with timeline awareness
Shell.scheduleNextUpdate(timelineContext) {
  // Store current timeline context for next cycle
  this.currentTimelineContext = timelineContext;
  
  // Use the bookmark action to set up next cycle
  if (this.pendingUpdates.length > 0) {
    // Process pending updates immediately
    this.processUpdate();
  } else {
    // Wait for next event or timer
    this.waitForNextEvent();
  }
}
```

## Key Loom Implementation Details

1. **DAG Event Storage**: Spaces store events in a directed acyclic graph structure, with each event referencing its parent.

2. **Timeline Context**: A data structure containing branch ID, parent references, and coherence status that flows through the system.

3. **Object State Management**: Spaces maintain separate object states for each timeline branch, projecting the appropriate state when needed.

4. **Timeline Coherence**: Spaces enforce rules preventing actions in decohered timelines, with special handling for primary timelines.

5. **Fork Creation**: A process that creates a new timeline branch, clones object states, and updates the event DAG.

6. **Context Projection**: Timeline-specific context is assembled

7. **External Propagation Control**: Activity Layer only propagates messages from the primary timeline to external systems.

This detailed sequence demonstrates how timeline management is primarily encapsulated in the Space layer, with other components remaining largely timeline-unaware. The Shell propagates timeline context without deeply understanding it, and Elements operate on timeline-projected state without timeline awareness.