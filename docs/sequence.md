# Detailed Processing Sequence: Message Flow from Activity Layer to Agent and Back

## Initial Conditions

**Current State:**
- Agent's Inner Space is "personal"
- External Space "orientation_hub" is already uplinked to the Inner Space
- Inner Space contains ContextManager Object
- Agent has previously exchanged messages with user "Kai"
- Shell is currently in Engagement Phase

> **Note**: This sequence document illustrates Model 2 (Uplinks to Remote Shared Spaces), where chat elements exist in remote spaces and are accessed through uplinks from the agent's Inner Space. This differs from Model 1 (Direct Chat Elements in Inner Space) described in the main components document. For details on both architectural approaches, please see the **[Multiuser Chat Participation Models](components.md#multiuser-chat-participation-models)** section in components.md.

## Sequence: Processing an Incoming Message

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
```

### 2. Activity to Space Mapping
```
// Activity layer translates external IDs to internal space/object IDs
SpaceMapping.resolveExternalIdentifier("telegram_adapter", "12345")
-> Returns: {spaceId: "orientation_hub", objectType: "ChatObject", objectId: "main_chat"}

// Event is transformed to internal format with timeline context (primary timeline)
InternalEvent {
  type: "new_message",
  spaceId: "orientation_hub",
  objectId: "main_chat",
  timelineContext: {
    branchId: "primary-branch-001",
    isPrimary: true,
    lastEventId: "evt-45678"  // Last event in this timeline
  },
  data: {
    sender: "Kai",
    content: "Have you had a chance to explore the ContextManager yet?",
    timestamp: "2025-03-14T15:22:47Z",
    externalId: "tg-msg-78901"
  }
}
```

### 3. Remote Space Event Processing
```
// Event is routed to the target remote Space
SpaceRegistry.getSpace("orientation_hub").receiveEvent(internalEvent)

// Remote Space processes the event
orientationHub.receiveEvent(internalEvent) {
  // Validate timeline coherence before processing
  if (!this.isCoherentTimeline(internalEvent.timelineContext)) {
    logger.error(`Timeline coherence violation: ${internalEvent.timelineContext.branchId}`);
    return ErrorResponse.TIMELINE_DECOHERENCE_ERROR;
  }
  
  // Generate event ID for the DAG
  const eventId = generateUUID();  // "evt-12346"
  
  // Add event to the remote space's DAG with parent reference
  this.eventDAG.addEvent({
    id: eventId,
    type: "object_event",
    objectId: internalEvent.objectId,
    data: internalEvent.data,
    timestamp: internalEvent.data.timestamp,
    parentId: internalEvent.timelineContext.lastEventId,
    branchId: internalEvent.timelineContext.branchId
  });
  
  // Update the timeline context with new last event
  const updatedContext = {...internalEvent.timelineContext};
  updatedContext.lastEventId = eventId;
  
  // Routes to appropriate object
  this.objects["main_chat"].receiveEvent(internalEvent.data)
  
  // Note: We don't create individual join event references in the Inner Space DAG
  // for each remote event. Instead, the agent's UplinkProxy maintains connection state
  // that tracks which events were received during the connection period.
  //
  // The UplinkProxy records:
  // 1. The connection start event (already recorded when the uplink was established)
  // 2. The range of events/timeline spans the agent has been exposed to
  // 3. Any disconnection events
  
  // Update UplinkProxy state with the new event
  const uplinkProxy = SpaceRegistry.getSpace("personal").getUplinkProxy("orientation_hub");
  uplinkProxy.trackRemoteEvent({
    remoteSpaceId: "orientation_hub",
    remoteEventId: eventId,
    timestamp: internalEvent.data.timestamp,
    objectId: internalEvent.objectId,
    timelineContext: updatedContext
  });
}
```

### 4. Chat Object Event Processing
```
// Chat object processes the message (timeline-unaware)
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
    changedAt: eventData.timestamp,
    timelineContext: updatedContext
  })
}
```

### 5. Shell Update Processing
```
// Shell receives notification of state change with timeline context
Shell.notifyStateChange(notification) {
  // Store the current timeline context
  this.currentTimelineContext = notification.timelineContext;
  
  // Checks current agent state
  if (this.currentPhase === "engagement") {
    // Initiates context update with timeline context
    this.updateContext(notification.timelineContext)
  } else {
    // Queues update for next phase transition
    this.pendingUpdates.push(notification)
  }
}

// Shell decides to trigger phase transition based on new message
Shell.updateContext(timelineContext) {
  // Transitions to contemplation phase
  this.currentPhase = "contemplation"
  
  // Prepares for context rendering with timeline context
  this.prepareContextRendering(timelineContext)
}
```

### 6. Context Rendering Preparation
```
Shell.prepareContextRendering(timelineContext) {
  // Collects spaces that need rendering
  const spacesToRender = [
    {spaceId: "personal", importance: "high"},  // Inner space (contains subjective timeline)
    {spaceId: "orientation_hub", importance: "high", timelineContext: timelineContext}  // Uplinked external space
  ]
  
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
  }
  
  // Begins rendering process with timeline context
  this.renderContext(spacesToRender, objectsToRender, timelineContext)
}
```

### 7. HUD Rendering Process with Timeline Awareness
```
Shell.renderContext(spaces, objects, timelineContext) {
  // Create context assembly buffer
  const contextAssembly = []
  
  // For each space
  for (const spaceInfo of spaces) {
    const space = SpaceRegistry.getSpace(spaceInfo.spaceId)
    
    // Determine if this space needs timeline context
    // Personal space uses its own subjective timeline, not the remote timeline context
    const spaceTimelineContext = (spaceInfo.spaceId === "personal") 
      ? null  // Subjective timeline used for Inner Space
      : timelineContext  // Remote timeline context used for uplinked spaces
    
    // Render space header info using Space delegate
    const spaceHeader = space.renderSpaceHeader({
      importance: spaceInfo.importance,
      forAgent: this.agentId,
      focusLevel: this.focus[spaceInfo.spaceId] || "normal",
      timelineContext: spaceTimelineContext
    })
    contextAssembly.push(spaceHeader)
    
    // For each object in the space
    for (const objInfo of objects[spaceInfo.spaceId]) {
      const object = space.getObject(objInfo.objectId)
      
      // For uplinked spaces, retrieve history based on connection spans
      let rendering
      if (spaceInfo.spaceId !== "personal" && objInfo.objectId === "main_chat") {
        // Get the UplinkProxy for this remote space
        const uplinkProxy = SpaceRegistry.getSpace("personal").getUplinkProxy(spaceInfo.spaceId);
        
        // Get connection spans (periods when agent was connected to this space)
        const connectionSpans = uplinkProxy.getConnectionSpans({
          limit: 5,  // Get the 5 most recent connection periods
          includeActive: true
        });
        
        // Get rendering with history from connection spans
        rendering = object.delegate.renderWithConnectionSpans({
          forAgent: this.agentId,
          importance: spaceInfo.importance,
          focusLevel: this.focus[spaceInfo.spaceId] || "normal",
          timelineContext: spaceTimelineContext,
          connectionSpans: connectionSpans
        })
      } else {
        // Normal rendering for Inner Space objects (uses subjective timeline)
        rendering = object.delegate.render({
          forAgent: this.agentId,
          importance: spaceInfo.importance,
          focusLevel: this.focus[spaceInfo.spaceId] || "normal",
          lastRendered: this.renderCache.getLastRenderedTime(object.getCacheKey())
        })
      }
      
      // Cache the rendering
      this.renderCache.store(object.getCacheKey(), rendering)
      
      // Add to context assembly
      contextAssembly.push(rendering)
    }
  }
  
  // Apply compression to the assembled context
  this.applyCompression(contextAssembly, timelineContext)
}
```

### 8. Chat Object Delegate Rendering with Connection Spans
```
// Detailed rendering process for remote chat object based on connection spans
ChatObjectDelegate.renderWithConnectionSpans(options) {
  // Retrieve history for all connection spans
  const historyBundles = [];
  
  // For each connection span, get the history
  for (const span of options.connectionSpans) {
    // Get history for this connection period
    const spanHistory = this.chatObject.getHistoryForTimelineSpan({
      startEventId: span.startEventId,
      endEventId: span.endEventId || "latest", // null for active connections
      timelineContext: options.timelineContext,
    });
    
    // Add to bundles with connection metadata
    historyBundles.push({
      messages: spanHistory,
      connectionStart: span.startTime,
      connectionEnd: span.endTime,
      isActive: span.isActive
    });
  }
  
  // Combine all messages across all bundles for rendering
  const allMessages = historyBundles.flatMap(bundle => bundle.messages);
  
  // Apply rendering for each message
  const renderedMessages = allMessages.map(msg => {
    // Find which connection span this message belongs to
    const parentBundle = this.findBundleForMessage(msg, historyBundles);
    
    return {
      type: "remote_message",  // Explicitly marked as remote
      id: msg.id,
      renderPriority: this.calculatePriority(msg, options),
      content: `<msg source="${options.remoteSpaceId}" username="${msg.sender}">${this.formatContent(msg.content)}</msg>`,
      metadata: {
        timestamp: msg.timestamp,
        sender: msg.sender,
        isNew: this.isNewForAgent(msg, options.forAgent),
        hasReactions: msg.reactions.length > 0,
        isRemote: true,  // Flag indicating this is from a remote space
        connectionStartTime: parentBundle.connectionStart,
        isActiveConnection: parentBundle.isActive
      }
    }
  });
  
  // If we have too many messages, create summary blocks for older spans
  if (renderedMessages.length > 20) {  // Arbitrary threshold
    // Group older messages into summary blocks by connection span
    const recentMessages = renderedMessages.slice(-15);
    const olderMessages = renderedMessages.slice(0, -15);
    
    // Create summaries for each older connection span
    const summarizedSpans = this.createConnectionSpanSummaries(olderMessages, historyBundles);
    
    // Return combined rendering
    return {
      type: "remote_chat_history",  // Explicitly marked as remote
      elements: [...summarizedSpans, ...recentMessages],
      metadata: {
        totalMessages: allMessages.length,
        summarizedCount: olderMessages.length,
        fullCount: recentMessages.length,
        lastMessageTime: allMessages[allMessages.length - 1].timestamp,
        remoteSpaceId: options.remoteSpaceId,
        connectionSpanCount: historyBundles.length,
        hasActiveConnection: historyBundles.some(b => b.isActive)
      }
    }
  }
  
  // If not too large, return all messages
  return {
    type: "remote_chat_history",  // Explicitly marked as remote
    elements: renderedMessages,
    metadata: {
      totalMessages: allMessages.length,
      summarizedCount: 0,
      fullCount: renderedMessages.length,
      lastMessageTime: allMessages[allMessages.length - 1].timestamp,
      remoteSpaceId: options.remoteSpaceId,
      connectionSpanCount: historyBundles.length,
      hasActiveConnection: historyBundles.some(b => b.isActive)
    }
  }
}
```

### 9. HUD Compression System

```
Shell.applyCompression(contextAssembly, timelineContext) {
  // First pass: Analyze context structure and relationships
  const contextAnalysis = this.analyzeContext(contextAssembly)
  
  // Determine current focus parameters
  const focusParams = {
    primarySpace: "orientation_hub",  // Space where action is expected
    activeObjects: ["main_chat"],     // Objects actively being interacted with
    semanticThemes: ["ContextManager", "exploration"] // Current conversational themes
  }
  
  // Determine global compression strategy based on:
  // 1. Available context window size
  // 2. Agent's current focus
  // 3. Semantic relevance of different elements
  // 4. Temporal factors (recency, causality)
  const compressionStrategy = this.determineCompressionStrategy(
    contextAnalysis, 
    focusParams,
    this.contextWindowLimit
  )
  
  // Apply compression policies to different elements
  const compressedAssembly = this.executeCompressionStrategy(
    contextAssembly,
    compressionStrategy
  )
  
  // Final assembly and presentation
  this.finalContext = this.assembleContext(compressedAssembly)
  this.presentContextToAgent()
}
```

### 9.1 Context Analysis for Compression

```
Shell.analyzeContext(contextAssembly) {
  // Build semantic graph of relationships between elements
  const relationshipGraph = new SemanticGraph()
  
  // Analyze each element in the context
  for (const element of contextAssembly) {
    // Extract semantic information
    const semanticInfo = this.extractSemanticInfo(element)
    
    // Add to relationship graph with metadata
    relationshipGraph.addNode(element.id, {
      type: element.type,
      source: element.source,
      renderPriority: element.metadata?.renderPriority || 5,
      recency: this.calculateRecency(element.timestamp),
      focusRelevance: this.calculateFocusRelevance(element),
      compressionHints: element.metadata?.compressionHints || {}
    })
    
    // Connect related elements
    if (element.metadata?.relatedElements) {
      for (const relatedId of element.metadata.relatedElements) {
        relationshipGraph.addEdge(element.id, relatedId, {
          relationship: element.metadata.relationships[relatedId] || "related"
        })
      }
    }
  }
  
  // Identify narrative chains, conceptual clusters, and causal sequences
  relationshipGraph.identifyPatterns()
  
  return {
    graph: relationshipGraph,
    temporalSpan: this.calculateTemporalSpan(contextAssembly),
    elementDensity: contextAssembly.length / this.calculateUniqueTopics(contextAssembly).length,
    importantElements: this.identifyKeyElements(relationshipGraph)
  }
}
```

### 9.2 Dynamic Compression Strategy

```
Shell.determineCompressionStrategy(analysis, focus, limit) {
  // Begin with lenient compression
  let compressionLevel = {
    global: 1,  // Base compression level (1-5)
    // Space-specific levels
    spaces: {
      "personal": 1,
      "orientation_hub": 1
    },
    // Object-specific levels within spaces
    objects: {
      "personal.context_manager": 1,
      "orientation_hub.main_chat": 1
    },
    // Semantic area levels
    semanticAreas: {}
  }
  
  // Calculate initial context size with this strategy
  let estimatedSize = this.estimateCompressedSize(analysis, compressionLevel)
  
  // Iteratively adjust compression until it fits
  while (estimatedSize > limit && !this.isMaxCompression(compressionLevel)) {
    // Identify areas to compress more aggressively
    const adjustments = this.identifyCompressionAdjustments(
      analysis, 
      compressionLevel, 
      focus,
      estimatedSize - limit
    )
    
    // Apply adjustments
    this.applyCompressionAdjustments(compressionLevel, adjustments)
    
    // Re-estimate size
    estimatedSize = this.estimateCompressedSize(analysis, compressionLevel)
  }
  
  // Special handling for remote history bundles - allow for partial compression
  if (analysis.hasRemoteHistoryBundles) {
    compressionStrategy.remoteHistoryHandling = {
      // For each space, determine how to handle its connection spans
      connectionSpanHandling: this.determineConnectionSpanCompressionStrategy(
        analysis.remoteHistoryBundles,
        focus,
        compressionLevel
      ),
      // Enable partial compression of bundles (message subsets within spans)
      allowPartialBundleCompression: true,
      // Specify how to handle spans at different temporal distances
      temporalCompressionPolicy: {
        recent: "preserve_detail",     // Keep recent spans detailed
        mid_term: "highlight_relevant", // Keep relevant messages from mid-term spans
        historical: "summarize"        // Summarize older spans
      }
    }
  }
  
  // Determine specific compression techniques for different element types
  return {
    levels: compressionLevel,
    techniques: {
      // Different techniques for different element types
      "space_header": this.selectTechniquesByType("space_header", compressionLevel),
      "chat_history": this.selectTechniquesByType("chat_history", compressionLevel),
      "remote_chat_history": this.selectTechniquesByType("remote_chat_history", compressionLevel),
      "object_state": this.selectTechniquesByType("object_state", compressionLevel),
      "knowledge_graph": this.selectTechniquesByType("knowledge_graph", compressionLevel)
    },
    // Elements that should remain uncompressed
    preserveVerbatim: analysis.importantElements,
    // How to handle cross-references during compression
    referenceHandling: this.determineReferenceStrategy(compressionLevel)
  }
}
```

### 9.3 Compression Techniques Across Element Types

The compression system applies different techniques depending on element type:

1. **Chat History**
   - Progressive summarization of conversations
   - Maintaining question-answer pairs intact where possible
   - Preserving emotional or decision-critical exchanges

2. **Knowledge Structures**
   - Abstracting detailed knowledge graphs to higher-level concepts
   - Collapsing similar information into representative examples
   - Maintaining structural relationships while reducing detail

3. **Space Information**
   - Condensing descriptive information while preserving navigational cues
   - Reducing member lists to key participants
   - Simplifying object listings based on relevance

4. **Object States**
   - Focusing on state changes relevant to current focus
   - Abstracting complex state to key parameters
   - Preserving action-relevant state information

5. **Cross-Element Compression**
   - Identifying redundant information across different elements
   - Maintaining causal chains while compressing independent events
   - Preserving narrative coherence across compressed elements

### 9.4 Delegate Compression Hints

Elements can provide hints about compression through their delegates:

```
// Example compression hints from a Chat Object delegate
{
  "compressionHints": {
    "narrativeChains": [
      {
        "elements": ["msg-123", "msg-124", "msg-125"],
        "importance": "high",
        "summary": "Discussion about ContextManager capabilities"
      }
    ],
    "keyElements": ["msg-122", "msg-125"],
    "compressibleBlocks": [
      {
        "elements": ["msg-118", "msg-119", "msg-120"],
        "suggestedSummary": "Initial greeting and platform introduction"
      }
    ],
    "relationshipPreservation": [
      {
        "primary": "msg-125",
        "related": ["msg-122", "msg-123"],
        "relationship": "builds_upon"
      }
    ]
  }
}
```

These hints inform but don't control the Shell's compression decisions. The Shell balances delegate hints against global context needs.

### 9.5 Remote History Bundle Specialized Compression

Remote history bundles are handled distinctly from the agent's subjective timeline, enabling specialized compression techniques:

```
// Example of how remote history bundles get special compression handling
Shell.compressRemoteHistoryBundles(remoteHistoryBundles, strategy) {
  // For each remote space's history bundles
  return remoteHistoryBundles.map(bundleSet => {
    // Get the compression strategy for this specific remote space
    const spaceStrategy = strategy.remoteHistoryHandling.connectionSpanHandling[bundleSet.remoteSpaceId];
    
    // Process each connection span (a period when the agent was connected)
    const processedSpans = bundleSet.connectionSpans.map(span => {
      // Determine temporal category of this span (recent, mid-term, historical)
      const temporalCategory = this.categorizeTemporal(span.connectionStart);
      
      // Apply appropriate compression based on temporal category
      switch (temporalCategory) {
        case "recent":
          // Recent spans - keep most detail
          return this.preserveDetailedSpan(span, spaceStrategy);
          
        case "mid_term":
          // Mid-term spans - keep thread integrity but compress less important messages
          return this.compressSpanPartially(span, spaceStrategy);
          
        case "historical":
          // Historical spans - heavy summarization
          return this.summarizeSpan(span, spaceStrategy);
      }
    });
    
    return {
      remoteSpaceId: bundleSet.remoteSpaceId,
      processedSpans: processedSpans
    };
  });
}

// Partial compression of a connection span - allowing precise control over which
// messages within a span are preserved, compressed, or summarized
Shell.compressSpanPartially(span, strategy) {
  // Group messages into conversational threads
  const threads = this.identifyConversationalThreads(span.messages);
  
  // For each thread, determine if it's relevant to current focus
  const processedThreads = threads.map(thread => {
    const relevance = this.calculateThreadRelevance(thread, strategy.focusTopics);
    
    if (relevance > strategy.highRelevanceThreshold) {
      // Keep important threads largely intact
      return {
        type: "preserved_thread",
        messages: thread.messages,
        compressionRatio: 1.0
      };
    } else if (relevance > strategy.lowRelevanceThreshold) {
      // Partially compress medium-relevance threads
      return {
        type: "partially_compressed_thread",
        // Keep essential messages (e.g., initial question, final answer)
        messages: this.extractEssentialMessages(thread.messages),
        // Generate summary of elided content
        summary: this.summarizeElided(thread.messages),
        compressionRatio: 0.4
      };
    } else {
      // Heavily summarize low-relevance threads
      return {
        type: "summarized_thread",
        // Just keep a generated summary
        summary: this.generateThreadSummary(thread.messages),
        compressionRatio: 0.1
      };
    }
  });
  
  return {
    spanId: span.id,
    connectionStart: span.connectionStart,
    connectionEnd: span.connectionEnd,
    processedThreads: processedThreads,
    overallCompressionRatio: this.calculateOverallRatio(processedThreads)
  };
}

Remote history bundles are unique in compression handling because:

1. **Span-Level Controls**: Each connection span (representing a period when the agent was connected to the space) can be compressed at different levels, with recent spans preserved in greater detail.

2. **Partial Bundle Compression**: Unlike most timeline events which are either kept or summarized entirely, remote history bundles can be selectively compressed - preserving some messages fully while summarizing others.

3. **Thread-Aware Compression**: The compression system preserves conversational thread integrity, keeping question-answer pairs together even when compressing surrounding context.

4. **Temporal-Contextual Balance**: Remote bundles use a combination of temporal factors (how long ago) and contextual factors (relevance to current focus) to determine compression levels.

5. **Cross-Span References**: The compression system preserves references between different connection spans to maintain narrative coherence across multiple connection periods.

This specialized handling enables much more efficient context usage while still maintaining the agent's ability to reference its experiences in remote spaces. Unlike subjective timeline events which represent direct experiences, remote history bundles can be treated more like "memories of observations" with variable precision.
```

### 10. Context Assembly and Presentation
```
Shell.assembleContext(compressedAssembly) {
  // Create system preamble
  const preamble = `<s>\nContemplation Phase\nThis is your space for reflection before engaging. You may organize your thoughts however you prefer.\n</s>`
  
  // Join all elements with appropriate separators
  const elements = compressedAssembly.map(element => {
    // Format based on element type
    if (element.type === "remote_chat_history") {
      // Clearly marked as remote history bundle
      return `<remote_chat history="bundle" space="${element.metadata.remoteSpaceId}">\n${
        element.elements.map(msg => msg.content).join("\n")
      }\n</remote_chat>`
    } else if (element.type === "chat_history") {
      // Regular chat history (from subjective timeline)
      return element.elements.map(msg => msg.content).join("\n")
    } else if (element.type === "space_header") {
      return element.content
    }
    // Handle other element types...
    
    return element.content
  })
  
  // Final formatted context
  return preamble + "\n\n" + elements.join("\n\n")
}

Shell.presentContextToAgent() {
  // Present the assembled context to the agent
  AgentInterface.sendContext(this.finalContext)
  
  // Update state to reflect waiting for agent
  this.currentState = "waiting_for_agent"
  this.lastContextUpdateTime = Date.now()
}
```

### 11. Agent Processing
```
// Agent (LLM) processes the context
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
  Shell.receiveAgentResponse(contemplationOutput)
}
```

### 12. Shell Processing Agent Response
```
Shell.receiveAgentResponse(response) {
  // Parse and store agent's notes
  this.parseAndStoreNotes(response)
  
  // Transition to engagement phase
  this.currentPhase = "engagement"
  
  // Prepare engagement options
  this.prepareEngagementOptions()
}

Shell.prepareEngagementOptions() {
  // Determine available actions for current space
  const currentSpace = SpaceRegistry.getSpace("orientation_hub")
  const availableActions = currentSpace.getAvailableActions(this.agentId)
  
  // Determine available Shell tools
  const availableTools = this.getAvailableTools()
  
  // Format options for presentation
  const engagementOptions = `
<system>
Engagement Phase
Current space: orientation_hub

Available Shell tools:
<sleep>seconds</sleep>
<timer>seconds</timer>
<bookmark>label</bookmark>

Available Space actions:
<msg>Send message to current space</msg>
<use>object_name</use>
  - Available objects: Information Panel, ContextManager, Interface Settings

<view>spaces</view>

You may combine ONE Space action with MULTIPLE Shell tools.
</system>
`

  // Send options to agent
  AgentInterface.sendContext(engagementOptions)
}
```

### 13. Agent Action Selection
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
  Shell.receiveAgentAction(selectedAction)
}
```

### 14. Shell Processing Agent Action
```
Shell.receiveAgentAction(actionString) {
  // Parse the action string
  const actions = this.parseActions(actionString)
  
  // Process each action
  for (const action of actions) {
    if (action.type === "space_action") {
      // Handle space action (msg in this case)
      this.processSpaceAction(action)
    } else if (action.type === "shell_tool") {
      // Handle shell tool (bookmark in this case)
      this.processShellTool(action)
    }
  }
  
 // Prepare for next cycle
  this.scheduleNextUpdate()
}

Shell.processSpaceAction(action) {
  if (action.name === "msg") {
    // Get current space
    const currentSpace = SpaceRegistry.getSpace("orientation_hub")
    
    // Find chat object
    const chatObject = currentSpace.getObject("main_chat")
    
    // Create message event
    const msgEvent = {
      type: "agent_message",
      sender: this.agentId,
      content: action.params.content,
      timestamp: new Date().toISOString(),
      replyTo: null
    }
    
    // Send to chat object
    chatObject.receiveEvent(msgEvent)
    
    // Send to activity layer for external propagation
    ActivityLayer.sendMessage({
      spaceId: "orientation_hub",
      objectId: "main_chat",
      message: msgEvent
    })
  }
  // Handle other action types...
}

Shell.processShellTool(action) {
  if (action.name === "bookmark") {
    // Create bookmark in agent's personal space
    const personalSpace = SpaceRegistry.getSpace("personal")
    const contextManager = personalSpace.getObject("context_manager")
    
    // Send bookmark event to context manager
    contextManager.receiveEvent({
      type: "create_bookmark",
      label: action.params.label,
      timestamp: new Date().toISOString(),
      context: {
        spaceId: "orientation_hub",
        objectId: "main_chat",
        referencedEvents: this.getCurrentContextEvents()
      }
    })
  }
  // Handle other tool types...
}
```

### 15. Activity Layer Propagation
```
ActivityLayer.sendMessage(msgData) {
  // Determine which external adapter(s) should receive this
  const mappings = SpaceMapping.getExternalMappings(msgData.spaceId, msgData.objectId)
  
  // For each mapping, format appropriately and send
  for (const mapping of mappings) {
    // Format for specific adapter
    const adapterSpecificMessage = this.formatForAdapter(mapping.adapterId, msgData)
    
    // Send to adapter
    AdapterRegistry.getAdapter(mapping.adapterId).sendMessage(adapterSpecificMessage)
  }
}

// Format specifically for Telegram adapter
ActivityLayer.formatForAdapter("telegram_adapter", msgData) {
  return {
    chat_id: 12345,  // From mapping
    text: msgData.message.content,
    parse_mode: "HTML",
    from_id: `agent_${msgData.message.sender}`
  }
}
```

### 16. Event Cycle Completion
```
// Telegram adapter processes the message
TelegramAdapter.sendMessage(message) {
  // Sends to Telegram API
  telegramClient.sendMessage(message.chat_id, message.text, {
    parse_mode: message.parse_mode
  })
  .then(result => {
    // Log success
    logger.info(`Message sent to Telegram chat ${message.chat_id}`)
    
    // Update message tracking
    this.sentMessages.set(result.message_id, {
      internal_id: generateUUID(),
      timestamp: new Date().toISOString(),
      sender: message.from_id,
      chat_id: message.chat_id
    })
  })
  .catch(error => {
    logger.error(`Failed to send message to Telegram: ${error.message}`)
  })
}

// Shell prepares for next cycle
Shell.scheduleNextUpdate() {
  // Use the bookmark action to set up next cycle
  if (this.pendingUpdates.length > 0) {
    // Process pending updates immediately
    this.processUpdate()
  } else {
    // Wait for next event or timer
    this.waitForNextEvent()
  }
}
```

## Sequence Summary

1. **External to Internal:** Activity layer converts external message to internal format with timeline context
2. **Remote Space Event Storage:** Message is stored in the remote space's DAG with proper timeline references
3. **Inner Space Reference:** A join event reference is created in the Inner Space DAG pointing to the remote event
4. **Object State Update:** Chat Object updates its state with the new message
5. **Shell Notification:** Shell is notified of state change with timeline context
6. **Rendering Preparation:** Shell prepares to render both Inner Space (subjective timeline) and remote space
7. **History Bundle Reconstruction:** Remote space history is reconstructed from join events during rendering
8. **Delegate Rendering:** Object Delegates render their state, with remote objects clearly marked
9. **Compression:** HUD applies compression policies to fit context window
10. **Context Assembly:** Full context is assembled with clear boundaries between subjective and remote content
11. **Agent Processing:** Agent processes the unified context and creates a response
12. **Action Selection and Processing:** Agent's actions are processed and propagated

This sequence demonstrates how the architecture maintains separate representations of subjective and objective history, while still providing a coherent unified context to the agent. The Inner Space contains only references to remote events (join events), which are used to reconstruct history bundles on demand during context building. This approach ensures memory efficiency while preserving the ability to access remote context when needed.



-------

