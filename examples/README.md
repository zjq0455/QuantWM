# Example inputs

- `robot_scene.png` and `prompt.txt`: the paired image/text example for
  LingBot and HY. The prompt is also used by the text-to-video backends.
- `matrix_scene.png`: an official Matrix-Game-2 demo first frame. Matrix
  generation is controlled by the image and actions, not the text prompt.
- `lingbot_actions/poses.npy` and `intrinsics.npy`: camera conditions for
  the included 93-frame example; `action_plan.json` describes the sequence.

These are illustrative inputs, not a calibration set or the paper's full
benchmark. You can replace them through CLI arguments without editing model
source. Keep each image, prompt and action sequence semantically matched.
Example assets do not change the licenses of upstream data or model weights.
