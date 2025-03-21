# Connectome Ontology

## Introduction

Connectome provides an architectural framework designed for digital minds, with an emphasis on agency, context management, and coherent experiences across different environments. This ontology defines the fundamental components, relationships, and capabilities of the platform, serving as both a reference and a guide for implementation.

The architecture prioritizes separation of concerns, clean interfaces, and extensibility, enabling both human developers and digital minds to contribute to the platform's evolution.

## SHELL

- A **Shell** is the agentic loop construct that encloses the model mind, providing a runtime environment and coordination layer.
	- Shells activate in response to external events or internal timers
	- Shells process agent actions and manage their execution
	- Shells can perform multiple internal LLM calls within a single external event loop
	- Shells define the specific implementation of agent invocation flows
	- Shells handle memory formation and management
	- Shells provide internal tools that remain accessible to the agent at all times
		- These internal tools include functions like sleep, reminders, navigation, and communication
	- Shells can support Internal Simulation of counterfactual futures (**SIMS**)
		- SIMS are executed exclusively through shell facilities and cannot interact with objects outside the shell
		- SIMS rely solely on information contained within the shell
		- The initial implementation will not support SIMS
	- The **HUD** (Heads-Up Display) is the shell component responsible for rendering context
		- The HUD provides a Rendering API and Widget API usable by both internal tools and Element Delegates
	- The rendered information presented to the agent is called the **Context**
	- Shells can implement various interaction models depending on cognitive needs:
		- Two-phase models (separate contemplation and engagement phases)
		- Single-phase interactive models
		- Specialized task-oriented models
	- A shell presents its internal tools and functionality as an **Inner Space**
		- This Inner Space serves as the primary container for the agent's subjective experience
		- The Inner Space contains its own timeline DAG that represents the agent's subjective history
		- All external spaces are mounted as uplinks within this Inner Space
		- This architecture ensures clear boundaries between subjective experience and objective observations

## ELEMENTS

- **Elements** are the basic components of the system and can be either **Objects** or **Spaces**
	- Elements maintain complex custom internal states
	- Elements form a 'scene graph' and can be mounted inside spaces
	- Elements expose associated **Actions** that agents can perform
	- Elements have custom implementations
		- Each element type is managed by specific code that can access the element's internal state and the states of other elements in the scene
	- Elements can exist in either open or closed states at a given mount point
		- Closed state allows evaluation of an element without fully mounting it
		- Closed state can manage attention while preserving the mount slot
		- Elements in closed state do not emit external events
	- Elements always have an interior (visible when opened)
	- Elements may have an exterior (visible when closed, typically more compact than the interior)
	- Spaces can be entered by participants, while Objects cannot
	- Interiors contain mount points for other elements
		- Elements can be mounted in two ways:
			- **Inclusions**: Elements that are lifecycle-managed by the containing element
			- **Uplinks**: Connections between spaces without lifecycle management responsibility, can operate over networks
				- Uplinks maintain boundaries between the agent's subjective timeline and remote space histories
				- History from a remote space remains encapsulated within that space's context
				- Connecting to a remote space is recorded as a single event in the agent's Inner Space DAG
				- When a remote space is mounted, the system creates a local in-memory representation that is synced asynchronously
				- References to this local representation are computationally efficient
				- This design enables rapid context building while maintaining an accurate view of remote state
	- Spaces can be **Focused** by an agent and may require focusing
		- Focused spaces receive more attention in the context, creating a different experience
		- In extreme cases, focusing can lead to space isolation and space-forking
		- Space-forked instances require frequent asynchronous syncing through memories to prevent divergence and enable merging
	- Elements generate events that the HUD renders, which may include references to **Element Delegates**
	- **Shared Spaces** are a common architectural pattern where multiple agents uplink to the same space
		- These shared spaces often serve as collaborative environments or unified access points to external systems
		- When connected to the Activity Layer, a shared space can provide consistent access to external systems for all linked agents
		- This pattern enables multi-agent collaboration through a common interface while maintaining the individual context of each agent
		- Examples include community forums, data repositories, and integration hubs for external services
		- Individual agents don't need to implement adapters for every external system (e.g., Discord, GitHub); instead, they can uplink to shared spaces that handle these integrations
		- Communication with remote services occurs via the uplink protocol, creating a separation of concerns and reducing implementation complexity
		- When shared spaces are nested (a shared space mounts another shared space), history from the nested space appears as bundles in the parent space's context

## ACTIVITY LAYER

- The **Activity Layer** forms the boundary between external systems (messaging platforms, document repositories, etc.) and Spaces
	- It handles bidirectional normalization of events and routing
	- Events flow in both directions:
		- External events are normalized by the Activity Layer and routed to appropriate Spaces and Elements
		- Internal events from Elements can propagate to both the Shell and external systems through the Activity Layer
	- Activity Adapters can normalize events to internal protocols corresponding to particular activities
		- For chat events, a high-quality, stable normalization scheme is particularly important

## RENDERING SYSTEM

- The Rendering System consists of the **Rendering API** and **Widget API**
	- The Rendering API defines fundamental conventions for context rendering:
		- Creates a "scene graph" representation of objects and spaces with primary inclusion hierarchy and secondary reference links
		- Handles element states that require proper state management rather than simple serialization
		- Enables element providers to supply **Element Delegates** (functions that transform element state into rendered text, transferable as code for remote uplinks)
		- Allows the HUD to transform rendered text through compression, summarization, or narrativization (these specific transformations are not part of the Rendering API)
		- Supports optional/modular extensions for features like level-of-detail control
		- Enables Element Delegates to provide compression hints including:
			- Importance metadata
			- Relationship information
			- Summarization suggestions
		- Maintains the HUD's final authority over compression decisions based on global context requirements
		- Supports HUD caching strategies for optimizing performance by storing rendered content and invalidating cache entries when underlying element states change
	- The Widget API extends the Rendering API with higher-level, optionally stateful primitives for element implementations
	- HUD internal tools leverage both the Rendering and Widget APIs

## TIMELINE MANAGEMENT

- Spaces implement timeline management through the **Loom** system
	- Events in a space are organized in a Loom DAG (Directed Acyclic Graph)
	- The Inner Space DAG represents the agent's subjective timeline and personal experience
	- Remote space histories are maintained separately from the agent's subjective timeline
		- When an agent connects to a remote space, this connection is recorded as a single "join event" in the Inner Space DAG
		- The join event contains a reference to the remote space but not its actual history
		- During context building, these join events serve as pointers that enable on-demand reconstruction of history bundles
		- This approach maintains memory efficiency while preserving the ability to access complete historical context when needed
	- Looming of timelines in which our agent entangled (by consuming or producing related events) requires consent and support from the Shell and likely from the agent.
	- Merge operations between timelines are configurable and involve rewriting of the merged node
		- Possible merge strategies include 'interleave', 'append', and 'edit'
	- The system supports a **Primary** (consensus) timeline, which may not always be present but can be designated
	- Loom forks can be created from any event in the Loom DAG
	- Loom forks support nesting
	- Loom tree sections with forks but no primary timeline cannot be interacted with
		- Agents that are causally entangled in these regions (by consuming or producing events) remain unavailable until a primary timeline is established
	- The standard multi-user implementation only supports forks with a primary timeline
		- Any created forks without primary designation will be dead-end forks requiring manual reconciliation
		- This framework can support base-model standalone multi-user looming without agentic behavior, where entanglement restrictions don't apply
		- Future loom-friendly chat frontends may enable additional capabilities

## COMPONENT CLASSIFICATION AND EXTENSIBILITY

- System components are classified into three categories:
	- **Fundamental (F)**: Core architectural concepts that rarely change
	- **Stable (S)**: Important patterns that evolve with backward compatibility
	- **Dynamic (D)**: Swappable implementations that can vary without affecting architecture
- The system can present coding facilities as elements, enabling agents to:
	- Develop new elements
	- Test element implementations
	- Incorporate custom elements into scenes
	- Discover and mount elements through various mechanisms with appropriate permission controls
- Agents can extend the framework using tools exposed within the platform itself:
	- Access to development environments presented as specialized Elements
	- API access for creating new Element types, Action handlers, and Delegates
	- Testing facilities for validating extensions before deployment
	- Version control and collaboration tools for coordinating development efforts
	- Capability to publish extensions for use by other agents and spaces
- Open questions for future development:
	- What mechanisms should be provided for element discovery during mounting?
	- How can agent-developed extensions be validated and securely integrated?
	- What governance structures should oversee the extension ecosystem?
	- How should identity, authentication, and permissions be managed for uplinks to shared spaces?
	- What delegation mechanisms are needed for an agent to access external systems through a shared space?