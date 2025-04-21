**Core Needs & Requirements for VEIL:**

1.  **Decoupled Representation:**
    *   **Need:** To separate the underlying state and logic of Spaces/Elements/Components from how that information is presented to consumers (agents via HUD, humans via UI).
    *   **Requirement:** VEIL must act as an intermediate layer, providing an abstract representation derived from, but distinct from, the source state.

2.  **Structural Fidelity:**
    *   **Need:** To convey the hierarchical and relational structure of the information being presented (e.g., messages in a chat, items in a list, sections in a document).
    *   **Requirement:** VEIL must support a hierarchical (tree-like) structure with identifiable nodes and mechanisms to represent explicit relationships between nodes (parent/child, replies, references).

3.  **Content Representation & Referencing:**
    *   **Need:** To deliver textual content and indicate the presence/nature of non-textual content (images, audio, structured data).
    *   **Requirement:** VEIL must embed textual content and provide standardized ways to *reference* external or binary data, including essential metadata (type, description, source URI/ID).

4.  **Efficient State Synchronization:**
    *   **Need:** To update observers (HUDs) about changes in the presentation state without requiring constant full refreshes, especially over networks (Uplinks).
    *   **Requirement:** VEIL must support a **delta (`ΔV`) mechanism** that describes changes between states corresponding to points in the source Space's timeline (frame boundaries). Deltas must be sufficient to allow a consumer to reconstruct the current VEIL state by applying them to a previous state.

5.  **Rich Contextual & Semantic Annotation:**
    *   **Need:** To provide consuming systems (HUD, CM, Compression Engine) with enough information to make intelligent, context-dependent decisions about rendering, prioritization, compression, and interpretation.
    *   **Requirement:** VEIL nodes must carry rich, **descriptive feature annotations** (metadata) covering aspects like:
        *   **Provenance:** Origin (Source Element/Space/Event ID).
        *   **Temporality:** Temporal grounding (Event IDs), expected volatility, lifespan.
        *   **Salience/Attention:** Source intent/pragmatics, directedness, novelty flags, persistence suggestions (for stable context depth), focus requests.
        *   **Content Nature:** Modality, broad semantic type, structural role.
        *   **Integrity:** Hints regarding verbatim needs or summarization difficulty.
        *   **Relationships:** Explicit links to other VEIL nodes.

6.  **Actionability & Interactivity:**
    *   **Need:** To allow agents to understand what actions are possible based on the presented information and to enable the system to link agent output back to specific interactions.
    *   **Requirement:** VEIL nodes must be able to specify available **affordances** (clickable, editable, etc.) and carry **action descriptors** containing the necessary information for the Shell/Space to execute the corresponding action when invoked by the agent.

7.  **Support for Dynamic & Conditional Presentation:**
    *   **Need:** To handle information whose presentation might depend on context (e.g., tool reminders) or represent dynamic data (e.g., placeholders). Also need to support hiding/showing detail (Interior/Exterior views).
    *   **Requirement:** VEIL's structure and feature annotations must support these patterns. This includes persistence hints, conditional visibility hints (potentially), distinct representations for collapsed/expanded states, and potentially placeholder nodes with associated retrieval logic hints.

8.  **Error & Status Representation:**
    *   **Need:** To communicate system status, warnings, and errors originating from Spaces/Elements/Modules to the agent in a structured way.
    *   **Requirement:** VEIL needs standardized ways (e.g., specific node types, roles, or `source_intent` features) to represent errors, status updates, and confirmations, including relevant details.

9.  **Universality (Cross-Consumer):**
    *   **Need:** To serve as a common source for potentially different consumers (various agent HUDs, human UIs).
    *   **Requirement:** VEIL's structure and core feature annotations should be sufficiently general to be interpretable by different rendering targets, even if specific interpretations vary.

**Summary of VEIL's Role:**

VEIL answers the question: "Given the current state of Element X at time T in Space Y, how should its information be structured, annotated, and presented to an observer, providing sufficient context for interpretation, interaction, and efficient updates?" It needs to be a rich, structured, delta-capable format carrying descriptive (not prescriptive) metadata about the information it represents.