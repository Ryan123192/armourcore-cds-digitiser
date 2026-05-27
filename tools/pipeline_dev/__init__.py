"""Pipeline-development dev scripts.

Lives alongside (not inside) the production pipeline code so it can
freely import / call into ``src/armourcore_cds/*`` without affecting
production behaviour.  Outputs of dev runs go to
``data/outputs/pipeline_dev/<step>/<timestamp>_<label>/``.

Add new dev scripts under ``tools/pipeline_dev/stepNN_*/`` — each is
independently runnable and never modifies pipeline code.
"""
