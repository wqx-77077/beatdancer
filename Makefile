PY ?= python3
OUT ?= dist

.PHONY: all demo video web test clean

all: demo

demo:
	$(PY) -m beatdancer.cli demo assets/demo_track.wav --out $(OUT) --servo-config

video:
	$(PY) -m beatdancer.cli render assets/demo_track.wav --out $(OUT)

test:
	$(PY) -m unittest discover tests -v

track:
	$(PY) tools/make_demo_track.py assets/demo_track.wav

clean:
	rm -rf $(OUT)/_frames $(OUT)/*.bvh $(OUT)/*.csv $(OUT)/choreo.json

board:
	$(PY) -m rdk.dance_service --mode replay --table $(OUT)/servo_table.csv --dry-run --speed 10
