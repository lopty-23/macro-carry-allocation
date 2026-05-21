# Macro Carry Allocation

> A long-only multi-asset allocation strategy combining trend and carry signals across nine ETFs, with monthly rebalancing, inverse-vol position sizing, and portfolio-level volatility targeting.

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

## Overview

This project extends Meb Faber's *A Quantitative Approach to Tactical Asset Allocation* ([SSRN 962461](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461)) by combining a 12-month trend signal with a cross-asset carry signal. The carry component is the key economic addition: where trend captures the behavioural under-reaction premium, carry captures a risk premium for bearing term, credit, and devaluation risk. The two signals are roughly uncorrelated within asset classes, so combining them diversifies rather than duplicating.

Full methodology and results to follow.

## Project Structure

(To be documented after build.)

## Data Sources

- **ETF prices, equity index P/E ratios, bond index yields, and 3M T-bill yields**: Bloomberg Terminal, exported manually to `data/raw/macro_carry_data.xlsx`. The raw file is excluded from version control because Bloomberg data is licensed.

## License

MIT