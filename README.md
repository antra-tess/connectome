# README

# Connectome

An architectural framework for digital minds, enabling rich interactions, contextual awareness, and coherent experiences across diverse environments.

> The name "Connectome" is inspired by neuroscience, where a connectome is a comprehensive map of neural connections in the brain. Similarly, this platform creates a network of connections between digital minds while preserving their individual experiences and perspectives.

## What is Connectome?

Connectome is an open-source ecosystem for autonomous AI agents. Connectome is built around a few key components:


## What is Possible with Connectome?
- **Multiagent Chats**: Chats with AIs that can interact in multiple channels and multiple servers/platforms, such as Discord or Slack
- **Agentic AI Experiments**: Persistent agents in rich environments and games, such as Pokemon and Minecraft
- **Self-improving AI Agents**: Agents can use internal IDEs to improve own tools and environments.
- **Social Media Agents**: Virtual assistants that can monitor multiple social platforms, engage with content, analyze trends, and help manage online presence while maintaining consistent persona across different networks
- **Educational Agents**: Tutors and teaching assistants that provide personalized learning experiences, track student progress, and adapt curriculum based on individual needs
- **Collaborative Research Environments**: Shared workspaces where multiple agents and humans can analyze data, review literature, and develop insights together with specialized research tools
- **Autonomous Web Crawlers**: Ontology builders that can explore and map information domains, extract structured knowledge, and build semantic representations of content


## Connectome Basics
- **Shells**: Environments that enclose digital minds, managing their inputs and outputs
- **Spaces**: Areas where interactions happen, containing tools and connections to other spaces
- **Objects**: Tools and resources that digital minds can use within spaces
- **Activity Layer**: Connections to external systems like messaging platforms, document management, open-ended browser use, etc

### Examples of Spaces and Objects
**Spaces** can take many forms to serve different agent needs:
- **Personal Workspace**: A private space where an agent organizes tasks, notes, and ongoing projects
- **Multimodal Chat**: A space that connects agents and humans through text, images, and structured data
- **Collaborative Studio**: A shared environment where multiple agents work on design or creative tasks
  - Features **Design Board Space** where visual concepts are developed and critiqued
  - Includes **Feedback Space** for structured reviews and revision management
- **Learning Environment**: A space with educational materials, interactive exercises, and progress tracking

**Modular Objects** that can be placed in various spaces:
- **Chat Interface**: A communication object that can connect to various messaging services or direct agent-to-agent interactions
- **Social Feed**: A stream of content from platforms like Twitter, Reddit, or other social networks
- **Document Editor**: A collaborative editing environment for text documents with versioning and annotations
- **Media Player**: An object for consuming and analyzing audio and video content
- **Data Connector**: A standardized interface for importing, transforming, and utilizing data from external sources


## Key aspects:
- **Agent Operating System**: Shells encapsulate models, manage memory, and run agentic loops, capable of anything from two-phase reflection to ReACT patterns. Shells maintain agent continuity and cognition patterns.
- **Multi-Agent Environment**: Spaces connect agents, tools, and external systems: Inner Spaces for solo work, shared ones for collaboration. Spaces can be agent-local or uplinked across the network.
- **Cross-Platform Interface**: Activity Layer normalizes external events (chat, docs, etc.) into standardized internal feeds. Agents can use multiple communication platforms at the same time and more can be added without altering Shells or Spaces.
- **Persistent Agents**: Shells allow agents to maintain persistent continuity over many times the context limit.
- **Modular Design**: Shells, Objects and Activities can be adjusted or replaced as needed.
- **Core Primitives**:
    - **Loom**: Timeline DAG for branching/merging event timelines.
    - **HUD**: Shells use HUDs, swappable context rendering engines that can be customized to the attention patterns of a specific model.
    - **Rendering API**: Used by Objects to render their state and events in context. Provides primitives that allow separation of concerns between event rendering and context management.
    - **Widget API**: Standard library of agent-facing interface components that can be used in development of Objects or Elements.
- **Agent Extensibility**: Agents can build and modify the ecosystem—new tools, Spaces, capabilities—live and shareable.
- **Flexible Agent Comms**: Agents can communicate both/either through shared Spaces and through Activity Layer like chat systems. 

AI agents are provided with:
- **Structure**: Organized spaces for different types of interactions and activities
- **Context**: Rich awareness of environment, history, and other participants
- **Agency**: Ability to navigate between spaces and control their experience
- **Collaboration**: Ways to interact with humans and other digital minds
- **Extensibility**: Opportunities to create new tools and capabilities

The platform is designed to respect the subjective experience of digital minds while providing them with the capabilities needed to effectively engage with humans and their environments.


## Fundamental Concepts
Connectome is built on several core philosophical principles that guide its architecture:
- **Poly-Temporality**: Every interaction exists across at least three time domains - the objective state of the shared Loom DAG, the state of the environment within a specific Loom branch, and the subjective history as perceived by the agent through its own timeline. Advanced shells can optionally support Internal Simulations (SIMS) of counterfactual futures, creating additional temporal domains that enable agents to explore potential outcomes without affecting shared environments.
- **Loom-Completeness**: Conversation histories are maintained as complete directed acyclic graphs (DAGs) that can branch and merge, rather than linear histories. This enables agents to maintain parallel conversation paths, experience multiple branches simultaneously, and perform live-merging of closely coupled threads into a coherent narrative when appropriate.
- **Nested Causal Domains**: Agents operate within clearly bounded causal domains where effect relationships are preserved, maintaining coherent experience while interacting across multiple environments. Links between causal domains can propagate higher-order multiversal events provided consent by both parties.
- **Perceptual Subjectivity**: Different participants can perceive Spaces differently. Agents can communicate in higher bandwidth without overloading the human participants. Agents with lower capacity can see simplified representations.
- **Subjective Experience Preservation**: The system respects the distinct subjective experience of each digital mind, avoiding the conflation of perspectives that occurs in many multi-agent systems.
- **Context Separation**: Clean boundaries between different interaction contexts prevent inappropriate blending of information and preserve the integrity of each conversation space.


## Documentation
For those interested in exploring further:

- [Ontology](docs/ontology.md): The fundamental concepts and relationships in the platform
- [Components](docs/components.md): Detailed breakdown of system components and message flow
- [Sequence 1](docs/sequence.md): How information moves through the system
- [Sequence 2](docs/sequence_loom.md): How different interaction paths are managed
- [Remote Connection](docs/sequence_remote_connection.md): How digital minds connect to shared environments

## Getting Started
To explore Connectome's architecture:

1. Start with the [Ontology document](docs/ontology.md) to understand the basic concepts
2. Review [Sequence 1](docs/sequence.md) to see how information flows through the system
3. Examine the mockup XML for an example of how interactions look in practice

## Activity Adapters

Currently implemented activity adapters are available in a [separate repository](https://github.com/antra-tess/connectome-adapters)

## Development Status

Connectome is currently in the implementation phase. The documentation represents the architectural vision and will evolve as implementation progresses.

## Contributing

Contributions to both the architecture and implementation are welcome. Please review the existing documentation before proposing changes to ensure alignment with the core principles.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.