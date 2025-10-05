# Roadmap

GPUX development roadmap and future plans.

---

## Vision

Make GPU-accelerated ML inference universal, simple, and accessible to everyone.

---

## Current Release: v0.1.0

✅ **Core Features**
- Universal GPU compatibility
- Docker-like CLI
- ONNX Runtime integration
- HTTP serving
- Comprehensive documentation

---

## Upcoming Releases

### v0.2.0 - Enhanced Providers (Q1 2025)

**Execution Providers**
- [ ] WebGPU support for browsers
- [ ] Vulkan backend
- [ ] Metal Performance Shaders (Apple)
- [ ] SYCL support (Intel oneAPI)

**Performance**
- [ ] Dynamic batching
- [ ] Model caching
- [ ] Quantization pipeline
- [ ] Mixed precision support

### v0.3.0 - Advanced Features (Q2 2025)

**Model Management**
- [ ] Model registry
- [ ] Version management
- [ ] A/B testing support
- [ ] Canary deployments

**Monitoring**
- [ ] Prometheus metrics
- [ ] OpenTelemetry integration
- [ ] Performance profiling
- [ ] Request tracing

### v0.4.0 - Enterprise Features (Q3 2025)

**Security**
- [ ] Authentication (API keys, OAuth)
- [ ] Model encryption
- [ ] Secure model storage
- [ ] Audit logging

**Scaling**
- [ ] Multi-GPU support
- [ ] Distributed inference
- [ ] Load balancing
- [ ] Auto-scaling

### v1.0.0 - Production Ready (Q4 2025)

**Stability**
- [ ] Production hardening
- [ ] Comprehensive testing
- [ ] Performance benchmarks
- [ ] Security audit

**Documentation**
- [ ] Enterprise deployment guide
- [ ] Video tutorials
- [ ] Case studies
- [ ] Best practices

---

## Feature Requests

### Most Requested

1. **WebGPU Support** - Run in browsers
2. **Multi-GPU** - Distribute across GPUs
3. **Model Registry** - Centralized model management
4. **gRPC API** - Alternative to HTTP
5. **Streaming Inference** - Real-time streaming

### Under Consideration

- Kubernetes operator
- Cloud marketplace images (AWS, GCP, Azure)
- GUI dashboard
- CLI plugins system
- Model compilation optimizations

---

## Platform Support

### Current
- ✅ Linux (x86_64, ARM64)
- ✅ macOS (Intel, Apple Silicon)
- ✅ Windows (x86_64)

### Planned
- [ ] Android (via Termux)
- [ ] iOS (via Pythonista)
- [ ] Raspberry Pi optimization
- [ ] NVIDIA Jetson optimization

---

## Community Priorities

Vote on features:
- [GitHub Discussions](https://github.com/gpux/gpux-runtime/discussions)
- [Feature Requests](https://github.com/gpux/gpux-runtime/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)

---

## How to Contribute

Help shape the roadmap:

1. **Share Use Cases** - Tell us how you use GPUX
2. **Vote on Features** - Upvote issues you care about
3. **Submit PRs** - Implement features
4. **Provide Feedback** - Share experiences

See [Contributing Guide](contributing.md) for details.

---

## Release Schedule

- **Minor releases**: Every 3 months
- **Patch releases**: As needed
- **Major releases**: Annually

---

## Long-Term Goals

### 2025
- Become the standard for ML inference
- Support all major GPU platforms
- 10,000+ GitHub stars
- Enterprise adoption

### 2026
- Cloud marketplace presence
- Managed GPUX service
- 100,000+ deployments
- Large-scale production usage

---

**Last Updated**: October 2024

For latest updates, see [Changelog](changelog.md).
