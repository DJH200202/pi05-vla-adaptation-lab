# Upstream code, models and data

This private portfolio is built on Physical Intelligence OpenPI and LIBERO. It does not claim authorship of their model architecture, pretrained weights, dataset, or simulator.

| Component | Reference |
|---|---|
| OpenPI | https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO/tree/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c |
| Raw demonstrations | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a |
| Base weights | gs://openpi-assets/checkpoints/pi05_base |

Each upstream component retains its own license and model/data terms. This repository grants no additional rights to those assets. Obtain assets from their original sources under the applicable terms. Full upstream repositories, data and checkpoints are not redistributed here.

Recorded simulation clips are outputs of the project's closed-loop evaluation. Their origin and SHA-256 are listed in demo/media-manifest.json. The demo is a recording viewer, not a live policy service.
