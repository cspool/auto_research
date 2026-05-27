# V-Rex

Local note: `/data3/paper_analysis/experiment_notes/kernel实验笔记/V-Rex_ Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval.md`

Relevance: video LLM runtime pipeline and custom hardware for irregular KV
retrieval.

Key method: Dynamic KV Cache Retrieval Engine with KVPU for bit-level clustering
and early-exit thresholding, plus KVMU for hierarchical KV memory and
cluster-wise mapping. KV prediction, fetch, attention, and FFN are overlapped.

Environment: custom V-Rex8/V-Rex48 accelerator models; AGX Orin and A100
baselines; DRAMSim3 and MQSim used in simulation.

Why it matters: video LLM concurrency is dominated by irregular selection and
data movement. V-Rex suggests this class may require hardware units rather than
GPU-only stream scheduling.

