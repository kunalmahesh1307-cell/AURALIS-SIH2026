# Embedded Deployment Roadmap

1. Freeze the validated DFN2 model/configuration.
2. Validate ONNX inference against the PyTorch reference.
3. Implement stateful streaming inference.
4. Move the inference pipeline to CM5.
5. Connect live audio input/output.
6. Measure per-frame processing time, latency, CPU, RAM and thermal behaviour.
7. Optimize only after correctness and streaming stability are established.
8. Integrate an I2S/codec-based audio front end.
9. Perform long-duration continuous tests.
