# Dependency provenance

The runtime repositories are Git submodules, following the same layout and workflow as CaP-X. The parent repository's gitlinks are the authoritative version pins; provision a fresh checkout with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

| Target | Upstream and pin | Runtime data |
| --- | --- | --- |
| `contact_graspnet_pytorch` | `https://github.com/uynitsuj/contact_graspnet_pytorch` (`main`) | Runtime weights live at `checkpoints/contact_graspnet/checkpoints/model.pt` inside the submodule |
| `sam3` | `https://github.com/Max-Fu/sam3.git` (`main`) | Runtime weights are provisioned at `checkpoints/sam3.pt` inside the submodule |
| `LIBERO-PRO` | `https://github.com/uynitsuj/LIBERO-PRO.git` (`master`) | simulator assets are tracked by the submodule |
| `libero_dependencies/robosuite` | `https://github.com/Max-Fu/robosuite` (`maxf/egl_context`) | model assets are tracked by the submodule |
| `openpi` | `https://github.com/Physical-Intelligence/openpi` (fixed commit in `openpi.commit`) | isolated Python 3.11 runtime; PyTorch transformers patch is applied inside that env |

PyRoKi is installed from the pinned Git revision in `pyproject.toml`, just as in CaP-X. `uv sync` installs all four local submodules editable. The bootstrap then runs `scripts/install_tool_checkpoints.py`, which provisions the two model files at their fixed repository-relative paths; the runtime launchers do not accept external checkpoint overrides.
