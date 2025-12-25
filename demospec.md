# Demo Spec

## Goal
Browser-based focus detection using webcam.

## Features
- Captures webcam frames in the browser
- Sends images to server for analysis
- Adjustable polling interval (0.5s - 10s slider)
- Displays FOCUSED or DISTRACTED status with reason

## Detection Logic
Uses a prompt to classify attention state:

**DISTRACTED** if:
- Phone visible
- Person not in frame
- Person not looking at camera

**FOCUSED** otherwise.
