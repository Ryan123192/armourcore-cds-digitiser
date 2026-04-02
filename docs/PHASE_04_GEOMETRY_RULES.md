# Phase 04 — Geometry Rules

## Goal
Improve rough vector output using rule-based geometric corrections.

## Intended rule families
- nearly straight -> straight
- nearly circular -> circle
- arc fitting where tangency is likely
- bilateral symmetry support
- simplification without obvious shape loss

## Principle
This stage should emulate the judgement that currently happens during manual Affinity cleanup, but in controlled, optional steps.
