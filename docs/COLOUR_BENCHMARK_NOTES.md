Colour benchmark harness notes
==============================

Input filenames must follow:
    <lighting>__<angle>__<sheet_state>__<capture_id>.jpg

Example:
    lowlight__goodangle__flat__01.jpg

This harness assumes a single A1 photo contains all four 260x350 colour-test sets:
    - top-left: set_magenta
    - top-right: set_cyan
    - bottom-left: set_orange
    - bottom-right: set_green

It does not try to detect the outer A1 page first. Instead it searches four overlapping
quadrant ROIs directly, which is simpler and more robust for this benchmark stage.
