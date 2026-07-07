# Checkpointing on Pathways with AxLearn

When implementing checkpointing or elastic snapshotting in AxLearn, it is critical to understand the architectural differences between traditional **Multi-Controller JAX** environments and the **Pathways** runtime. Treating Pathways like a standard JAX cluster leads to severe layer violations, session collisions, and ultimately, failed runs.

## 1. Multi-Controller JAX vs. Pathways

### Multi-Controller JAX (e.g., Cloud TPU VM setups)
In a standard multi-controller JAX setup, the user's application code acts as the distributed orchestrator. 
- **Explicit Cluster Management:** You manually manage network connectivity and worker membership.
- **Explicit Initialization:** You must explicitly initialize and shut down the distributed cluster using `jax.distributed.initialize()` and `jax.distributed.shutdown()`.
- **Manual Teardown:** You often manage sockets and teardown logic explicitly (e.g., using `atexit` handlers or manual socket cleanup) to ensure workers sync up and gracefully shut down when preemptions occur.

### Pathways
Pathways abstracts away cluster management and orchestration, acting as a unified operating system for ML accelerators.
- **Implicit Cluster Management:** The **controller manages the mesh**. The runtime natively handles data sharding, collective operations, and data movement.
- **Automatic Initialization:** Distributed initialization and shutdown are managed **automatically** by Pathways.
- **Infrastructure Teardown:** Network connectivity, memory cleanup, and worker health are handled strictly by the runtime layer, not the application layer.

## 2. Checkpointing Principles for Pathways

When designing checkpointing/snapshotting mechanisms for Pathways in AxLearn, adhere to the following principles to maintain clean abstraction layers:

### A. Avoid Redundant Initialization and Shutdown
**Do not** call `jax.distributed.initialize()` or `jax.distributed.shutdown()`. Since Pathways manages the cluster topology, redundant calls will conflict with Pathways' internal initialization workflow, causing session collisions, crashing the C++ proxy client, and fighting the runtime. Always wrap these calls with `if not utils.is_pathways_proxy():` if the code must run in both environments.

### B. No Manual Socket or Network Management
**Do not** attempt to manage TCP sockets, custom shutdown signals, or sidecar networking scripts. Parsing `/proc/net/tcp` to manually close leaked sockets, or pinging worker daemons on port 29001 manually, is a severe layer violation. Pathways inherently manages the network layer.

### C. Remove `atexit` Handlers for Cleanup
Avoid manual memory and network cleanup hooks (`atexit`). These handlers often mask true failures, interfere with Python teardown, and prevent the Pathways runtime from executing its native elastic recovery and teardown procedures. 

### D. Use JAX-Native Health Detection (In-Band)
Instead of relying on out-of-band socket pings or TCP connection checks to determine worker health, use **JAX operations**. A simple dummy JAX collective operation (like a small addition or synchronization step) naturally leverages the Pathways mesh. If a worker is dead, the collective op will fail/timeout cleanly, allowing the controller to detect preemption organically without breaking abstraction layers.

### E. Include Grain in Snapshot State
For deterministic resumption, the data iterator's state must be preserved exactly. Ensure that the **Grain** state is explicitly included in the snapshot state alongside the trainer state. This guarantees that data loaders resume from the precise point of preemption without dropping or repeating data batches.

### F. Avoid Intra-Process Recursive Restarts
Do not attempt to catch errors, tear down object state, and dynamically instantiate a new `SpmdTrainer` within the same step loop. A clean infrastructure-level restart is always preferred over an application mutating itself mid-flight.

### Summary
Pathways is designed to handle the heavy lifting of distributed orchestration. The cleanest and most robust checkpointing implementations are those that rely entirely on native JAX primitives (for logic and in-band health checks) and trust the Pathways runtime to manage the hardware, network, and distributed session.
