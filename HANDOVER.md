# BodySync: handover

Final presentation materials, code and results for BodySync, a personal stress score from a wristband.
Team: Swayam Lakhotia, Ananya Khaitan, Bhavya Bajaj. Code: https://github.com/swaylakh/bodysync

## The final deliverables

| What | File |
|---|---|
| Final slides (15 slides) | `docs/slides/BodySync_Final.pptx` (PDF copy: `docs/slides/BodySync_Final.pdf`) |
| Final report (5 pages) | `report/BodySync_Final_Report.pdf` |
| Report source (LaTeX) | `report/BodySync_Final_Report.tex` + `figures/fig7_example_person.png` |
| Interactive 3D walkthrough | `docs/interactive/BodySync_3D.html` (open in a browser; needs internet for the 3D library) |
| Presenter guide | `docs/presenter_guide.md`, with detailed notes per person in `docs/presenter_notes/` |
| Full technical write-up | `README.md` |

To rebuild the report PDF: `pdflatex BodySync_Final_Report.tex` inside `report/` (or upload the .tex and `figures/fig7_example_person.png` to Overleaf).

## Folder map

```
README.md            full technical write-up: method, every result, how to run
HANDOVER.md          this file
docs/
  slides/            final deck (+ archive/ with the earlier weekly decks)
  interactive/       the 3D walkthrough page
  presenter_guide.md, presenter_notes/
  Paper_Summary.pdf, Related_Papers.md   background reading
report/
  BodySync_Final_Report.tex / .pdf       the final report
  main.tex, overleaf.tex, tables/, refs.bib, make_tables.py   the earlier (mid-term) report
  archive/           older progress report
src/                 all code (data loaders in src/datasets/, final system in src/final_eval.py)
results/             every result as CSV (iteration_log.csv = every idea tried, final_test.csv = locked test)
figures/             figures used in the reports
data/                not included, see below
```

## Datasets (not included)

The raw data is about 7 GB and each public dataset has its own licence, so it is not uploaded here.
Download links and where each one goes are in `README.md` under "Getting the data"; our own dataset
(ZU-PWD '25) is described in `data/DATASET_SOURCE.md`.

| Dataset | Used for | Source |
|---|---|---|
| ZU-PWD '25 | the first idea (sync) | IEEE DataPort, doi:10.21227/2tth-wj12 |
| WESAD | lab stress | https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx |
| Stress-Predict | lab stress | https://github.com/italha-d/Stress-Predict-Dataset |
| Stress + exercise | lab stress | https://physionet.org/content/wearable-device-dataset/1.0.1/ |
| CAN-STRESS | everyday life | https://zenodo.org/records/14842061 |
| AffectiveROAD | real driving | https://www.media.mit.edu/groups/affective-computing/data/ |
| Nurse stress, ADARP | tried, left out (4 test people each) | see README |

## Reproducing the results

```
pip install -r requirements.txt
python src/datasets/<name>.py        # once per downloaded dataset
python src/final_eval.py --final     # frozen system on the locked 40% of people
python src/new_datasets.py           # frozen system on real driving
```
Every other run (rounds 1–17, checks, plain-formula baseline) is listed in `README.md` under "Running it".

## Limitations (also in the deck, slide 14, and the report)

- Everyday life rests on 10 hidden people and driving on 9.
- In lab studies stress always comes at the same point, so timing can't be fully ruled out there; the driving test is the main evidence against it.
- Everyday stress is milder and rated a few times a day; calm ratings cluster in the evening.
- Each person's normal uses their whole recording; a real app would only have the past.
- It separates stressful stretches from calm ones but doesn't follow moment-to-moment stress.
- All stress data comes from one wristband family (Empatica E4).
