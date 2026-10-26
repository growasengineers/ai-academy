# GrowAsEngineers AI Academy: companion code

Runnable, tested code, exercises and solutions for the GrowAsEngineers AI Academy video lessons on YouTube. Every number shown in a lesson is computed by the code here, and the tests prove it.

## Lessons

| Lesson | Track | Module | Folder |
|---|---|---|---|
| How Machines Learn from Examples, Part 1: Loss, Slopes and Gradient Descent by Hand | Classical Machine Learning | C01 How Models Learn from Examples | [`lessons/C01-L01-how-models-learn`](lessons/C01-L01-how-models-learn) |
| How Machines Learn from Examples, Part 2: Gradient Descent in Code, Learning Rates and Testing | Classical Machine Learning | C01 How Models Learn from Examples | [`lessons/C01-L02-gradient-descent-in-practice`](lessons/C01-L02-gradient-descent-in-practice) |

## Running a lesson

```bash
cd lessons/<lesson-folder>/code
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q          # the tests behind every number in the video
```

Each lesson's `code/README.md` lists its Python and package versions and what to run. Try `exercises.md` before opening `solutions.md`.

No lesson needs an API key or a GPU unless its README says so.

Code is MIT licensed (see `LICENSE`).

More lessons are added as they are published. growasengineers.com
