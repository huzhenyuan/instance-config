# instance-config

ComfyUI instance provisioning config repository.

Each GPU instance clones this repo on startup and runs `setup.py <group_id>`
to install custom nodes and download model files.

## Structure

```
groups/               # per-group provisioning config (nodes + models)
  face_detection_swap_shared.yaml
  face_combine_group.yaml
  image_to_video_group.yaml
  video_edit_group.yaml
  video_face_swap_group.yaml
workflows/            # ComfyUI workflow JSON templates
  w1a.json  ...
setup.py              # entrypoint: python3 setup.py <group_id>
```

## Usage

```bash
git clone <this-repo> /opt/gpus/instance-config
python3 /opt/gpus/instance-config/setup.py face_detection_swap_shared
```

## Adding a new group

1. Create `groups/<group_id>.yaml` with `nodes:` and `models:` lists.
2. Add the group to `src/config.yaml` in the scheduler.
# instance-config
# instance-config
