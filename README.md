# Überblick

Dieses Verzeichnis beinhaltet den Code zum Bachelorprojekt von Anina Klaus, mit dem Thema Complex Word Identification im Deutschen. Diese Aufgabe ist als Teil einer Pipeline zur Übersetzung in Leichte Sprache gedacht, die im Gegensatz zu end2end-Ansätzen auf hohe Nachvollziehbarkeit und Kontrolle der einzelnen Schritte setzt.

Die verschiedenen Klassifikatoren sind im Verzeichnis `cwi/` abgelegt. Zusätzlich gibt
es im Verzeichnis `notebooks/` zwei `jupyter-notebook`-Dateien,
in denen die Funktionsweise der Klassifikatoren vorgeführt wird. Dies sind `sharedtask_data_tests.ipynb`
und `mdr_data_tests.ipynb`.

Die verwendeten Daten finden sich im Verzeichnis `data/`, sodass die notebook-Dateien direkt ausgeführt werden können.
Dies umfasst die Daten der CWI Shared Task von 2018, abrufbar unter
https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset.html,
sowie den aus MDR-Nachrichtentexten erstellte Datensatz.

Eine weitere jupyter-notebook-Datei (`create_cwi_data.ipynb`) führt den beschriebenen Vorgang der Korpus-Erstellung
aus Paralleltexten vor.

# Setup

Um die zuvor genannten Dateien ausführen zu können, müssen die in requirements.txt aufgelisteten Packages
installiert sein. Dies kann zum Beispiel wiefolgt mithilfe von miniconda geschehen:

1. miniconda installieren (https://docs.conda.io/projects/miniconda/en/latest/)
2. Zum Projektordner navigieren
3. Ein conda-Environment erstellen: `$ conda create -n cwi python=3.9`
4. Das Environment aktivieren: `$ conda activate cwi`
5. Die Requirements installieren: `$ pip install -r requirements.txt`
6. Zusätzlich wird ein spacy-Model für das Deutsche benötigt: `python -m spacy download de_core_news_md`

Anschließend kann der Befehl `jupyter-lab` ausgeführt werden, um jupyter zu starten.
In der jupyter-lab-Umgebung können nun die notebook-Dateien ausgeführt werden.

# Allgemeines

- Anina Klaus
- Bachelorarbeit Computerlinguistik
- Department Linguistik, Humanwissenschaftliche Fakultät
- Universität Potsdam

Potsdam, 04.09.2023

