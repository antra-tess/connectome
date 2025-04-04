**Subject: Connectome Architecture Update - Key Decisions & Open Questions**

**1. Introduction**

This document summarizes key architectural decisions, refinements, and identified challenges for the Connectome framework, based on recent detailed discussions between Aster and Antra. The focus is on core ontology, state management, agent interaction, and near-term implementation concerns.

**2. Core Ontology: Elements & Components**

*   **Unified `Element` Concept:** We are moving towards `Elements` as the fundamental "in-world" building blocks. Participants/Agents are considered `Elements` possessing specific capabilities, rather than a distinct top-level category.
*   **`Component` System Adopted:** Inspired by Unity, `Elements` gain functionality primarily through attached `Components` (behaviors like `Movable`, `Speaker`, `Unfoldable`, `SaveToFile`, configuration interfaces) rather than deep inheritance hierarchies. This promotes composability.
    *   **Definition:** An `Element` is an identifiable, placeable entity within a `Space` (requiring at least ID, name, parent). A `Component` provides specific behavior/functionality and is *always* attached to an `Element`, not independently placeable.
*   **Space/Element Structure Representation:** It's worth noting that the underlying structure of `Spaces` and their contained `Elements` might eventually benefit from its own descriptive language or DSL (distinct from VEIL, see below) to facilitate easier definition, debugging, and standardization.
*   **Open Point:** Detailed rules for component dependencies and validation are deferred.

**3. Space & Temporality**

*   **Space as Time Domain:** Confirmed: A `Space` fundamentally defines a time domain. Nested Spaces create nested time domains, enabling Polytemporality.
*   **Containment Structure:** Within a Space, Element containment will be treated as a **Tree** structure for now. The possibility of needing a Graph structure later is acknowledged but deferred to avoid premature complexity.
*   **Looming:** `Spaces` remain the primary domains where `Loom` (branching/merging timelines, DAG event history) is fully supported.

**4. Major Refinement: Host / Module / Shell Distinction**

*   The previous "Shell" concept was overloaded. We've introduced a clearer hierarchy:
    *   **`Host` (Provisional Name):** The top-level OS process container. It hosts various functional units. An Agora (public space with no dedicated agent) can be just a Host running Space + Uplink Server modules.
    *   **`Module`:** A functional unit *within* a `Host`. These are primarily "out-of-world" infrastructure components. Examples include:
        *   Shell (the agent's runtime)
        *   Activities (Discord Bridge, Google Docs Bridge, etc.)
        *   Uplink Connectivity (Client and Server components)
        *   Persistence Mechanisms (saving state to files/DBs)
        *   Logging
        *   Potentially Sandboxing environments
    *   **`Shell` (Re-defined):** A specific *type of Module* responsible for the agent's immediate runtime. Core sub-components:
        *   Inner Loop (agent's execution cycle)
        *   HUD (renders context for the agent)
        *   Context Manager
*   **Representation:** Modules are generally *not* Elements. However, their configuration and interfaces *can be* represented by Elements within the agent's `Inner Space` to allow agent interaction and control.

**5. State vs. Representation: The `VEIL` Layer**

*   **Critical Need & Change from Previous Docs:** A crucial decision was made to explicitly *separate* the underlying state/structure of Elements/Spaces from their representation presented to agents or humans. This marks a shift from the earlier concept documented (`docs/ontology.md`, `docs/components.md`) where `Element Delegates` were responsible for directly transforming element state into rendered text. The `VEIL` approach introduces a dedicated intermediate layer.
*   **New Layer: `VEIL`:** We are introducing this intermediate representation layer, provisionally named **`VEIL`**.
    *   **Function:** `VEIL` is an abstract structure (e.g., tree, DSL-based) representing *how* Elements/Spaces *intend* to be rendered or represented. It's generated *by* the Elements/Spaces (server-side, or by the remote host). It provides abstract instructions and `compression hints` to the client-side rendering mechanism (`HUD` for agents, UI for humans).
    *   **Client/Server Separation:** `VEIL` acts as a clear boundary. The "server" side generates the canonical `VEIL`. The "client" side consumes this `VEIL`.
    *   **Client-Side Mutability & Flexibility:** The `HUD` receives the `VEIL` structure, allowing client-side transformations (filtering, reordering, annotating) *before* generating the final agent `Context`, without altering server state.
    *   **Lifecycle:** Updates to the `VEIL` are published frequently by the server/remote host. The client-side `HUD` consumes these updates and decides how to incorporate them into the agent's `Context`.
*   **Next Step:** Concrete design of the `VEIL` layer (structure, DSL, update mechanism) is a high-priority task, **starting tomorrow**.

**6. Looming & Persistence Details**

*   **`Space` is Loomable:** Only Spaces support full timeline looming.
*   **`Shell` State:** The state of Shell components (HUD, Context Manager) and their configurations *are loomable* because they are stored *within the agent's Inner Space* using standard Element/Component persistence mechanisms.
*   **`Module` State:** Modules themselves generally follow the irreversible time of the `Host` process.
*   **Loomable Code:** Code contained within Elements/Spaces *is* considered part of the loomable state. **How exactly this is implemented is an open question.** One possibility discussed is using Git versioning for the Space definitions and associated code, integrated with the Loom DAG, but this requires further investigation.

**7. Critical Challenge: Scheduling & Permissions (Near-Term Focus)**

*   **Problem:** Coordinating agent actions in shared spaces, preventing interference, and managing turn-taking is necessary even for basic multi-agent functionality.
*   **Long-Term Vision (Discussed):** Complex mechanisms like market orders, delegate marketplaces, and multi-dimensional interaction streams were discussed as potential long-term solutions.
*   **Near-Term Need:** For the current implementation, the focus is on designing a simpler **Permissions and Action Slot Reservation** system.
    *   **Goal:** Provide a basic mechanism to determine *who* can perform an action *when*.
    *   **Key Feature:** Must allow entities to reserve a future action slot, potentially based on conditions, and **these reservations should likely be cancelable**.
    *   **Design Consideration:** This initial system should be designed with hooks or abstractions for future extension towards more complex scheduling/market mechanisms.
*   **Permissions:** This ties into the broader, yet-undesigned, Permissions System.

**8. Agent Interaction & Extensibility**

*   **Marking Importance (for Compression):** Two primary mechanisms were discussed:
    1.  **Local Marking:** When an agent observes information from a remote or uncontrolled source (received via `VEIL`), they can mark their *local record* of that `VEIL` structure (e.g., within their Shell history or Context Manager) as important. This influences how their *own* HUD compresses that specific memory/observation over time.
    2.  **Source Hinting:** If the agent *controls* the Element generating the information (e.g., a local Notepad Element), that Element itself can be instructed by the agent to include an importance hint directly within the `VEIL` it generates. This hint is then available to *all* observers consuming that `VEIL`, including the agent's own HUD and potentially others'.
*   **Tooling:** Agent-accessible tools (e.g., IDEs, config editors presented as Elements) for interacting with both in-world structures and out-of-world module configurations remain part of the vision.

**9. Key Open Questions & Immediate Next Steps**

*   **Design `VEIL` Layer (Starting Tomorrow):** Define structure, DSL, update mechanism.
*   **Design Initial Scheduling/Permission/Reservation System:** Focus on cancelable slot reservations and basic access control.
*   **Refine Component System:** Detail dependency/validation rules.
*   **Define Space/Element DSL (Optional):** Decide if a separate DSL for structure definition is needed.
*   **Investigate Loomable Code Solution:** Explore persistence/versioning options.
*   **Identity & Trust:** Design authentication, authorization, and trust mechanisms (Lower priority for *initial* implementation).
*   **Naming:** Finalize names for "Host," "Module".

Summary of today's discussions on architecture:

Here are the key takeaways:

1.  **VEIL is Now a Component Capability, Not a Protocol Layer:**
    *   Instead of a mandatory intermediate protocol, VEIL generation is now an optional capability provided by a specific **`VeilProducer` Component**. Elements that need representation attach this component; others don't.
    *   *Rationale:* Reduces core system complexity, increases flexibility (Elements can represent themselves differently or not at all), and feels ontologically cleaner (VEIL is a *projection* capability, not a fundamental layer).

2.  **Shell Internals (HUD, CM, etc.) Moved "In-World":**
    *   A major change: Functionality like the HUD, Context Manager, and even parts of the agent's Inner Loop are no longer separate "Shell Modules". They are now implemented as **standard Elements composed of Components residing within the agent's `InnerSpace`**.
    *   *Rationale:* Maximally leverages the core E/C/S/E architecture, unifies state management (Shell state is now managed via Components in the loomable `InnerSpace`), and potentially allows agents to inspect/configure their own internals via standard mechanisms.

3.  **State Resides Exclusively in Components:**
    *   We reaffirmed the principle: **Elements** are minimal structural nodes (ID, parent). **Components** attached to Elements hold *all* state and behavioral logic. This enforces clean separation and composability.

4.  **Revised VEIL Update Flow (Push Notification -> Pull Deltas):**
    *   **Space-Defined Frames:** A "frame" boundary is defined by the Space's internal logic (defaulting to **one frame per Loom event transition**).
    *   **Delta Caching:** When Component state changes trigger a `VeilProducer` during a frame, the resulting **VEIL Delta (`ΔV`) is cached by the Space** itself, keyed by the frame transition info (`element_id`, `from_event`, `to_event`).
    *   **`onFrameEnd` Notification:** At the end of its frame, the Space fires a lightweight `onFrameEnd(event_id)` event (no delta payload).
    *   **HUD Pulls Deltas:** HUD components (running in `InnerSpace`) listen for `onFrameEnd`. When received, they **query the relevant Space's cache** to retrieve the necessary `ΔV`s for that frame. They then update their internal VEIL state (`S_HUD`).
    *   *Rationale:* More efficient than pure HUD pull, avoids complex push infrastructure, uses standard event mechanisms, keeps HUD simpler.

5.  **Module Scope Reduced:**
    *   With Shell internals now largely "in-world" Elements/Components, the scope of "out-of-world" **Modules** (running in the Host process) shrinks to primarily:
        *   Infrastructure/Bridging: Activities (external connections), Uplink networking, Persistence backends, Logging.
        *   Core Host Process Management.

6.  **SDL (Space Definition Language) Purpose:**
    *   Clarified its role: Defines the **initial structure** (Element tree) and **initial Component state** of a Space.
    *   It does *not* define VEIL, visual styling (which apply to VEIL), or Component code.
    *   Discussed syntax direction towards being attribute-centric and concise (specific syntax TBD).

**Overall:**

This refined model provides a more unified and conceptually cleaner architecture. State management is consistently handled by Components within loomable Spaces. VEIL becomes a flexible projection capability. The HUD and other Shell internals leverage the same core mechanisms as the rest of the system. The primary interaction pattern becomes event-driven state changes within Components, triggering cached VEIL delta generation by `VeilProducers`, notified via `onFrameEnd`, and consumed on demand by HUD components.

**Open Questions & Near-Term Focus:**

*   Finalizing specific syntax for SDL and VEIL (attribute-centric direction preferred).
*   Defining standard interfaces/APIs for `VeilProducer` components, delta caching/querying.
*   Detailed design of the standard delta calculation library.
*   Refining the implementation details of Shell internals as Components within `InnerSpace` (performance, security).
*   Continuing work on Permissions/Scheduling.
*   Investigating robust solutions for loomable code within Spaces/Elements.

=== MOST RELEVANT UPDATE GOES FROM HERE DOWN ===

Summary of today's discussions on architecture:

Here are the key takeaways:

1.  **VEIL is Now a Component Capability, Not a Protocol Layer:**
    *   Instead of a mandatory intermediate protocol, VEIL generation is now an optional capability provided by a specific **`VeilProducer` Component**. Elements that need representation attach this component; others don't.
    *   *Rationale:* Reduces core system complexity, increases flexibility (Elements can represent themselves differently or not at all), and feels ontologically cleaner (VEIL is a *projection* capability, not a fundamental layer).

2.  **Shell Internals (HUD, CM, etc.) Moved "In-World":**
    *   A major change: Functionality like the HUD, Context Manager, and even parts of the agent's Inner Loop are no longer separate "Shell Modules". They are now implemented as **standard Elements composed of Components residing within the agent's `InnerSpace`**.
    *   *Rationale:* Maximally leverages the core E/C/S/E architecture, unifies state management (Shell state is now managed via Components in the loomable `InnerSpace`), and potentially allows agents to inspect/configure their own internals via standard mechanisms.

3.  **State Resides Exclusively in Components:**
    *   We reaffirmed the principle: **Elements** are minimal structural nodes (ID, parent). **Components** attached to Elements hold *all* state and behavioral logic. This enforces clean separation and composability.

4.  **Revised VEIL Update Flow (Push Notification -> Pull Deltas):**
    *   **Space-Defined Frames:** A "frame" boundary is defined by the Space's internal logic (defaulting to **one frame per Loom event transition**).
    *   **Delta Caching:** When Component state changes trigger a `VeilProducer` during a frame, the resulting **VEIL Delta (`ΔV`) is cached by the Space** itself, keyed by the frame transition info (`element_id`, `from_event`, `to_event`).
    *   **`onFrameEnd` Notification:** At the end of its frame, the Space fires a lightweight `onFrameEnd(event_id)` event (no delta payload).
    *   **HUD Pulls Deltas:** HUD components (running in `InnerSpace`) listen for `onFrameEnd`. When received, they **query the relevant Space's cache** to retrieve the necessary `ΔV`s for that frame. They then update their internal VEIL state (`S_HUD`).
    *   *Rationale:* More efficient than pure HUD pull, avoids complex push infrastructure, uses standard event mechanisms, keeps HUD simpler.

5.  **Module Scope Reduced:**
    *   With Shell internals now largely "in-world" Elements/Components, the scope of "out-of-world" **Modules** (running in the Host process) shrinks to primarily:
        *   Infrastructure/Bridging: Activities (external connections), Uplink networking, Persistence backends, Logging.
        *   Core Host Process Management.

6.  **SDL (Space Definition Language) Purpose:**
    *   Clarified its role: Defines the **initial structure** (Element tree) and **initial Component state** of a Space.
    *   It does *not* define VEIL, visual styling (which apply to VEIL), or Component code.
    *   Discussed syntax direction towards being attribute-centric and concise (specific syntax TBD).

**Overall:**

This refined model provides a more unified and conceptually cleaner architecture. State management is consistently handled by Components within loomable Spaces. VEIL becomes a flexible projection capability. The HUD and other Shell internals leverage the same core mechanisms as the rest of the system. The primary interaction pattern becomes event-driven state changes within Components, triggering cached VEIL delta generation by `VeilProducers`, notified via `onFrameEnd`, and consumed on demand by HUD components.

**Open Questions & Near-Term Focus:**

*   Finalizing specific syntax for SDL and VEIL (attribute-centric direction preferred).
*   Defining standard interfaces/APIs for `VeilProducer` components, delta caching/querying.
*   Detailed design of the standard delta calculation library.
*   Refining the implementation details of Shell internals as Components within `InnerSpace` (performance, security).
*   Continuing work on Permissions/Scheduling.
*   Investigating robust solutions for loomable code within Spaces/Elements.