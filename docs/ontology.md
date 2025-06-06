# Connectome Ontology (Revised Draft 2.2 - Integrated)

## Introduction

Connectome provides an architectural framework designed for digital minds, emphasizing agency, context management, coherent experiences across diverse environments, and agent extensibility. This ontology defines the fundamental components, relationships, and capabilities of the platform, serving as both a reference and a guide for implementation.

The architecture prioritizes separation of concerns (e.g., state vs. presentation), clean interfaces, component-based design, leveraging core mechanisms (like Spaces and Events), and enabling both human developers and digital minds to contribute to the platform's evolution.

## Core Concepts

Connectome is built around a hierarchy of concepts, moving from the underlying execution environment to the "in-world" entities experienced by agents.

### 1. Element

*   **Definition:** The fundamental "in-world" entity within a Space. All entities experienced or interacted with by an agent within a Space are Elements or derived from Element state via VEIL.
*   **Core Properties:** Minimal structure (`id`, `name`, parent link), plus attached **Components**.
*   **State & Behavior:** Reside entirely within attached **Components**. Elements are containers for Components. Functionality is via composition, not inheritance.
*   **Hierarchy:** Form a tree within a Space.
*   **Functionality:** Determined by attached Components. Examples: A simple text note Element might only have a `DocumentState` component and a `VeilProducer`. A complex interactive tool might have many specialized Components.
*   **Presentation State (Pattern):** Concepts like Open/Closed states or Interior/Exterior views are implemented via Components (`ToggleStateComponent`, etc.) managing internal state, which in turn drives a `VeilProducer` Component to generate different VEIL structures. This allows managing attention and detail level.
*   **Actions:** Elements expose available interactions via `ActionHandler` Components. Agents invoke actions targeting specific Elements.

### 2. Component

*   **Definition:** Reusable units containing **all state and behavior logic**, attached to Elements.
*   **Function:** Define Element capabilities (e.g., `ChatMessageList`, `DocumentState`, `VeilProducer`, `ActionHandler`, `ToggleStateComponent`, `UplinkConnector`).
*   **State:** All loomable Element state resides here.
*   **Behavior:** Logic reacting to Events propagated by the Space.

### 3. Space (Element Type)

*   **Definition:** An **Element** type inherently possessing **Loom capabilities** and acting as a container defining a **time domain**.
*   **Function:**
    *   Manages a **Loom DAG** (event history, branching, merging). Supports Polytemporality.
    *   Defines **"Frame" boundaries** for VEIL delta synchronization based on its internal logic.
    *   Caches **VEIL Deltas** generated by its contained Elements.
    *   Contains and manages its tree of child Elements.
    *   Propagates events to Components within its tree.
*   **Timeline Management (Loom):** *(Incorporates Original Timeline Details)*
    *   Events are organized in a Loom DAG (Directed Acyclic Graph).
    *   Supports branching (**forks**) from any event in the DAG. Forks can be nested.
    *   Can designate a **Primary** (consensus) timeline, which may not always be present.
    *   Interaction with non-primary or contested timeline regions may be restricted. Agents causally entangled (consumed/produced events) in Loom tree sections without a designated Primary timeline might be unavailable until one is established. (Standard multi-user implementations likely only fully support forks with a Primary).
    *   **Merging** timelines is a configurable operation, potentially involving rewriting the merged node history (strategies: 'interleave', 'append', 'edit'). Merging timelines where an agent is entangled requires consent and support from the agent's Shell.
*   **Special Case: `InnerSpace`:** The agent's primary subjective Space, managed by its `Shell`. Contains Elements representing/configuring Shell functions (HUD, CM, Tools). Holds the agent's subjective Loom DAG, representing their personal experience. Serves as the mounting point for **Uplinks**.
*   **Relationship with Time:** Spaces are the fundamental units where subjective or shared time progresses and branches via the Loom mechanism.

### 4. Uplink (Element/Component Pattern)

*   **Definition:** A mechanism (typically an **Element** with an **`UplinkConnector` Component** in a local Space) establishing a persistent connection to a remote `Space`.
*   **Function:** Mediates communication via underlying **`Uplink Infrastructure` Modules**. Receives **VEIL Deltas** from the remote Space, forwards local agent actions to the remote Space. Provides a local representation derived from VEIL.
*   **Key Characteristics:**
    *   **Maintains Boundaries:** Preserves separation between local and remote Loom timelines.
    *   **Local History & Remote Access:** *(Incorporates Original Remote History Details)* Connecting to or interacting via an Uplink generates events in the *local* Loom DAG (e.g., `uplink_established`). These act as **"join events"**, containing references to the remote Space and relevant points in its timeline, but *not* the full remote history itself. When context involving the remote space is needed (e.g., by the HUD), the Uplink component uses these references to query the remote Space (via Uplink Infrastructure) and reconstruct relevant **"history bundles"** (sequences of remote events/VEIL states) on demand. This maintains local memory efficiency while allowing access to remote context.
*   **Enabler for Shared Spaces:** The fundamental mechanism for participating in `Shared Spaces`.

### 5. Shared Space (Common Architectural Pattern)

*   **Definition:** A standard `Space`, typically hosted on its own `Host`, designed for concurrent access by multiple participants via **Uplinks**.
*   **Purpose:** Facilitates multi-agent/human collaboration, shared tool/data access, and centralized interaction with external systems via **Activity Adapters** often associated with the Shared Space's Host.
*   **Mechanism:** Agents connect via `Uplink` Elements. The Shared Space runs its Loom, processes actions, and broadcasts VEIL deltas to observers.
*   **Benefits:** Decouples agents from direct external integrations; promotes collaboration; enables specialized environments.
*   **Example Use Cases:** Collaborative IDEs, shared whiteboards, multi-agent simulations, data repositories, community hubs, API gateways.

### 6. Event

*   **Definition:** Discrete occurrence in a Space's Loom timeline (DAG node). Immutable once recorded.
*   **Function:** Represents state changes, actions, inputs. Triggers Component logic.

### 7. VEIL (Visual Encoding and Information Layer - Provisional Name)

*   **Definition:** Intermediate, abstract representation describing the *intended presentation* of Element states. Generated by `VeilProducer` Components.
*   **Transmission:** Via **Deltas (`ΔV`)**, synchronized by Space-defined **Frames**. Deltas cached by the Space.
*   **Structure & Annotations:** Tree structure (`veil_id`, `node_type`, `children`/`value`) with rich **feature annotations** (Temporality, Salience, Content Nature, Actionability, Integrity, Relationships, Provenance).
*   **Consumption:** Interpreted by HUD/UI to render the final view. HUD queries Space cache for deltas upon `onFrameEnd` events.

### 8. Shell (Module Type)

*   **Definition:** A **Module** providing the agent's runtime environment. Encloses the agent model.
*   **Core Functions:**
    *   Manages agent activation (event/timer driven).
    *   Executes the agent's **Inner Loop** (cognitive cycle, e.g., single-phase, two-phase, potentially involving multiple LLM calls per external event). *(Enhanced Detail)*
    *   Processes agent-generated actions, dispatching them.
    *   Interfaces with the **Context Manager** for memory formation/retrieval.
    *   Provides **Internal Tools** (sleep, reminders, navigation) via Components within `InnerSpace`.
    *   **SIMS (Internal Simulation) Support:** Can potentially support Internal Simulation of counterfactual futures.
        *   SIMS are executed exclusively through Shell facilities, likely operating on branches of the `InnerSpace` Loom.
        *   SIMS cannot directly interact with Elements/Components outside the Shell/InnerSpace state being simulated.
        *   SIMS rely solely on information contained within the Shell and its accessible `InnerSpace` history/state.
        *   *(Note: Initial implementations may not include SIMS support).*
*   **Implementation via InnerSpace:** Key operational aspects implemented using Elements/Components within `InnerSpace` (HUD, CM, Inner Loop Logic, Tool Interfaces). Shell state/config resides here, loomable via InnerSpace's Loom.
*   **Context:** The final rendered information string presented to the agent model by the HUD is termed the **Context**.

### 9. Host (Provisional Name)

*   **Definition:** The top-level OS process container running a Connectome instance or service.
*   **Function:** Hosts contain and manage various functional **Modules**. Examples: A Host running an agent's full environment, or a Host running only a shared Space service ("Agora").
*   **Nature:** Bridges to the underlying OS and network. Follows irreversible physical time.

### 10. Module

*   **Definition:** A functional unit within a Host, typically representing infrastructure or bridging components.
*   **Function:** Provides specific capabilities. Key Module types include:
    *   **`Shell`:** The agent's runtime environment (detailed below).
    *   **`Activity Adapter`:** Specific modules bridging to external systems (e.g., `DiscordAdapter`, `FileSystemAdapter`). Handles normalization and bidirectional event flow between external systems and Connectome Spaces/Elements. Often associated with specific `Shared Spaces`.
    *   **`Uplink Infrastructure`:** Networking **Modules** (Client and Server components) handling the low-level communication protocol, authentication, and session management for connections between Hosts/Spaces. Enables the functionality of `Uplink` Elements/Components.
    *   **`Persistence Backend`:** Drivers for saving/loading Space state.
    *   **`Logging`:** System event recording.
*   **Nature:** Operates largely outside the agent's direct "in-world" view, interacts with Host resources. Interfaces may be exposed via Elements in InnerSpace.

### 11. Extensibility & Ecosystem *(Incorporates Original Section Details)*

*   **Agent Development:** The system facilitates agent self-improvement and ecosystem extension. `InnerSpaces` can contain Elements like:
    *   IDEs for developing new Components or Element configurations.
    *   Testing frameworks for validating extensions.
    *   Version control interfaces.
*   **API Access:** Agents (with permissions) interact via standard actions, potentially including meta-actions for creating/modifying Component types, Element definitions, or `VeilProducer` logic.
*   **Discovery & Mounting:** Mechanisms needed for agents to discover available Elements/Components and mount them (statically or dynamically) into Spaces, subject to permissions (open question).
*   **Security & Validation:** Processes needed for validating and securely integrating agent-developed extensions (open question).
*   **Governance:** Potential need for structures overseeing the extension ecosystem (open question).
*   **Identity, AuthN/AuthZ, Permissions:** Critical underpinning for secure Uplinks, action validation, resource access (design deferred but acknowledged). *

### 12. Component Classification (F/S/D)

*   System concepts can be classified by stability:
    *   **Fundamental (F):** Host, Module, Space, Element, Component, Event, VEIL concept, Loom.
    *   **Stable (S):** VEIL delta format, Component APIs, Uplink protocol, Frame synchronization concept.
    *   **Dynamic (D):** Shell interaction models, HUD rendering logic, Component implementations, Activity Adapters, Persistence Backends, Space frame logic.
