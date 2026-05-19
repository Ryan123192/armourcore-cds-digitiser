"""ArmourCore CDS Vectoriser — GUI package.

Thin GUI wrapper around the existing image-processing pipeline.  All
processing logic lives in ``src/armourcore_cds/``; this package only
provides:

* A pipeline runner that calls the existing phases sequentially and
  reports progress via callbacks (``pipeline_runner.py``).
* A Qt worker thread that runs the pipeline off the UI thread
  (``worker.py``).
* The main Qt window (``main_window.py``).
* The app entry point (``app.py``).

Deleting this folder reverts the project to command-line-only operation
without affecting any pipeline behaviour.
"""
