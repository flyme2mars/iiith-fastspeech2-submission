# checkpoints/

Trained weights do not go in this top-level folder.

- English: `English/checkpoints/{step}.pth.tar`
- Telugu: `Telugu/checkpoints/{step}.pth.tar`

`scripts/train.py` writes those paths and resumes from the newest step. `*.pth.tar` is gitignored. If a file is over GitHub's limit, use Git LFS or put a durable download link in the root README.
