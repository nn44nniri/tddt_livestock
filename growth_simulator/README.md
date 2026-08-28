# LiGAPS-Beef Python Translation

A Python translation of the LiGAPS-Beef modeling workflow, including the herd-level integrated model and two standalone sub-models for thermoregulation and feed intake/digestion.

## Summary

This repository contains a working Python translation of the LiGAPS-Beef modeling framework originally described in a three-paper series by Van der Linden and colleagues. LiGAPS-Beef is a mechanistic, daily time-step model for simulating potential and feed-limited beef production, quantifying yield gaps, and identifying the biophysical factors that define and limit cattle growth.

The code bundle includes:

- `LiGAPSBeef20180301_herd_worked.py`: integrated herd-level LiGAPS-Beef workflow.
- `Thermoregulation_submodel_20170922_worked.py`: standalone thermoregulation sub-model and sensitivity-analysis style driver.
- `Feed_digestion_submodel20170907_worked.py`: standalone feed intake and digestion sub-model data/formalism implementation.
- `FRACHA19982012.csv`: weather or climate input data used by the herd workflow.
- the three attached reference papers describing the model, sub-model evaluation, and whole-model evaluation.

This code should be understood as a faithful Python conversion of an originally published R implementation, not as the canonical original source. The scientific basis follows the published papers, while the software includes some practical Python-oriented additions such as NumPy arrays, path handling, optional progress display, and non-interactive plotting.

## Introduction

### Why LiGAPS-Beef exists

LiGAPS-Beef was developed to estimate the biophysical scope to increase beef production across production systems and regions. It is based on production ecology concepts that distinguish three production levels:

1. **Potential production**: determined by genotype and climate only.
2. **Feed-limited production**: constrained by feed quality and available feed quantity.
3. **Actual production**: the realised production under real-world constraints.

The difference between potential or feed-limited production and actual production is the **yield gap**. In this framework, LiGAPS-Beef is used to quantify that gap and to identify the defining and limiting factors behind it.

### What the model simulates

LiGAPS-Beef integrates three interconnected sub-models:

1. **Thermoregulation**
2. **Feed intake and digestion**
3. **Energy and protein utilisation**

Together they simulate growth, feed intake, energy use, heat stress, cold stress, protein limitation, digestion limitation, and herd-level feed efficiency.

### What is in this repository

The repository is centered around the integrated herd script and supported by two standalone sub-model files:

- The **integrated herd model** is intended for illustration cases and herd-level outputs.
- The **thermoregulation script** isolates thermal balance and sensitivity-analysis logic.
- The **feed digestion script** isolates feed composition, rumen/intestine digestion, and feed-related parameter structures.

## Scientific Scope

The code is designed to support the same broad scientific questions addressed in the source papers:

- How much beef production is possible under ideal climate-genotype conditions?
- How much does feed quality or feed quantity reduce that potential?
- Which biophysical factor is defining growth on a given day?
- Which factor is limiting growth on a given day?
- How do climate, breed, housing, and diet affect feed efficiency at animal and herd level?

## Repository Contents

### Main files

#### `LiGAPSBeef20180301_herd_worked.py`
Integrated LiGAPS-Beef herd workflow. It contains:

- model illustration logic for selected cases,
- sensitivity-analysis scaffolding,
- herd aggregation,
- tracking arrays for defining and limiting factors,
- graph generation,
- feed efficiency summaries.

This script is the central executable research script in the attached code set.

#### `Thermoregulation_submodel_20170922_worked.py`
Standalone thermoregulation script implementing the thermal-balance logic. It includes:

- weather sensitivity sweeps,
- breed-specific thermal parameter libraries,
- daily heat balance calculations,
- lower and upper critical temperature behavior,
- outputs for thermal heat-release components.

#### `Feed_digestion_submodel20170907_worked.py`
Standalone feed intake and digestion script containing:

- feed composition arrays,
- digestion-rate and passage-rate parameterization,
- fill-unit related structures,
- energy and protein related feed descriptors.

#### `FRACHA19982012.csv`
Climate/weather input file used by the integrated model for at least part of the illustration workflow.

## Model Architecture

LiGAPS-Beef is organized around three interconnected mechanistic blocks.

### 1. Thermoregulation sub-model

This block computes whether an animal is within its thermoneutral zone or under cold/heat stress. It represents the animal as a three-layer cylinder:

- body core,
- skin,
- coat.

The daily heat balance includes five major heat-flow pathways:

1. latent and convective heat release from respiration,
2. latent heat release from the skin,
3. long-wave radiation balance of the coat,
4. convective heat losses from the coat,
5. solar radiation intercepted by the coat.

The output is a heat-balance interpretation used by the rest of the model:

- under cold conditions, additional energy is required,
- under hot conditions, feed intake may be reduced and extra energy may be spent on panting,
- within the thermoneutral zone, normal metabolic allocation can proceed.

### 2. Feed intake and digestion sub-model

This block simulates feed intake and digestive conversion based on:

- feed composition,
- available feed quantity,
- fill units,
- digestion capacity,
- animal energy requirements.

The formulation is based on:

- the **INRA fill unit system**, and
- the **Chilibroste et al. rumen model**.

The model distinguishes the major feed constituents used in the papers:

- SNSC,
- INSC,
- DNDF,
- UNDF,
- SCP,
- DCP,
- UCP.

It produces outputs such as:

- feed intake,
- metabolisable energy (ME),
- digestible protein.

### 3. Energy and protein utilisation sub-model

This block is embedded in the integrated herd workflow. It partitions available energy and protein over:

- maintenance,
- physical activity,
- growth,
- gestation,
- lactation.

It converts the sub-model outputs into animal growth and beef production, while also returning heat-production signals back to the thermoregulation block.

## Challenge Questions

This repository is best understood through the scientific and computational challenges it addresses.

### Challenge 1: How can we estimate beef production potential across regions and systems?
**Answer:** By simulating cattle growth mechanistically with genotype and climate as defining factors, and by comparing potential and feed-limited scenarios with actual production conditions.

### Challenge 2: How can the model identify why growth is limited?
**Answer:** By recording the biophysical factor that is most defining or limiting on each day. In the integrated code, tracking arrays are used for genotype limitation, heat stress, cold stress, digestion-capacity limitation, energy deficiency, and protein deficiency.

### Challenge 3: How can climate effects be represented without full CFD or sub-daily simulation?
**Answer:** By using a daily thermoregulation model adapted from within-day thermal models. The code uses daily weather inputs and computes whether heat balance is feasible under minimum and maximum heat-release conditions.

### Challenge 4: How can feed quality and feed quantity both be represented mechanistically?
**Answer:** By combining fill-unit constraints with digestive conversion and passage-rate logic, allowing the model to limit intake by digestion capacity, diet composition, or physical feed availability.

### Challenge 5: How can animal-level results be translated into herd-level performance?
**Answer:** By simulating productive and reproductive animals separately and aggregating their outputs to herd-level summaries such as total production, total intake, and herd feed efficiency.

### Challenge 6: How can one model serve both illustration and sensitivity analysis?
**Answer:** The attached code contains a case-illustration workflow and also sensitivity-analysis scaffolding, especially for weather and parameter perturbations using a one-at-a-time approach.

## Answers to the Main Challenges

### A. Defining factors
In LiGAPS-Beef, defining factors are primarily:

- genotype,
- climate.

Genotype affects thermal behavior and growth potential. Climate affects whether growth can proceed without stress. The code tracks these influences explicitly.

### B. Limiting factors
The main limiting factors are:

- feed quality,
- available feed quantity.

These are represented through digestion capacity, energy deficiency, and protein deficiency.

### C. Why daily time step matters
The daily time step makes the model computationally practical and compatible with available weather and herd-management data. It also means the model represents average daily behavior rather than intra-day behavior.

### D. Why herd-level feed efficiency matters
Animal-level growth does not automatically translate to efficient herd-level production. LiGAPS-Beef therefore aggregates feed intake and beef production at herd scale, allowing evaluation of system-level efficiency and breeding/management scenarios.

## Formalisms

This section summarizes the main conceptual and mathematical structures behind the code.

### 1. Production ecology formalism
The model follows a hierarchy:

- **Potential production** = f(genotype, climate)
- **Feed-limited production** = f(genotype, climate, feed quality, feed quantity)
- **Yield gap** = potential or feed-limited production − actual production

### 2. Thermal balance formalism
At a conceptual level, the thermoregulation sub-model enforces a daily energy balance:

**Heat production + absorbed solar load = heat release by respiration + sweating + long-wave radiation + convection + rain evaporation + reflected solar radiation**

The model evaluates:

- minimum heat release,
- maximum heat release,
- whether additional cold-related energy is required,
- whether feed intake and growth must be reduced to avoid overheating.

### 3. Digestion formalism
Feed digestion is represented through digestion and passage of major feed fractions in the rumen and intestines. The code follows the paper’s structure where:

- feed fractions are digested or passed onward,
- total digested dry matter corresponds to digestible energy (DE),
- metabolisable energy (ME) is derived from DE,
- digestible protein is also derived from the feed-composition pathway.

A key relation stated in the source material is:

- **ME ≈ 0.82 × DE** for cattle.

### 4. Feed-intake constraint formalism
The effective feed intake is determined by the most restrictive of several bounds, including:

- intake required to meet energy demand,
- climate-constrained intake,
- digestion-capacity-limited intake,
- available-feed-limited intake.

### 5. Energy partitioning formalism
Net energy and protein are allocated among maintenance, physical activity, growth, gestation, and lactation. Growth is converted into tissues and ultimately beef production.

### 6. Herd aggregation formalism
The herd script scales from individual animals to the herd unit by combining:

- reproductive animals,
- productive animals,
- culling/replacement style probabilities,
- herd-level totals for production and intake,
- feed efficiency calculations.

## Inputs

Typical inputs reflected in the papers and visible in the code include:

- breed-specific parameters,
- generic cattle parameters,
- physical and chemical constants,
- daily weather data,
- housing information,
- feed types and feed composition,
- available feed quantities,
- diet composition,
- slaughter-weight or case-specific settings.

## Outputs

Typical outputs include:

- total body weight (TBW),
- feed intake,
- beef production,
- feed efficiency,
- minimum and maximum heat release,
- cold stress and heat stress indicators,
- digestion limitation indicators,
- energy and protein deficiency indicators,
- herd-level aggregate summaries,
- illustration graphs.

## Illustration Cases

The integrated herd script contains explicit illustration-case arrays and comments that link the code to the published illustration exercise. In the source paper, the model is illustrated for Charolais and Brahman × Shorthorn cattle in France and Australia.

In the attached Python version, the integrated script still carries those illustration-oriented settings, including arrays for:

- genotype,
- direct climate input,
- housing regime,
- feed availability,
- diet number,
- slaughter weight,
- defining/limiting factor traces.

This means the repository is especially suitable for:

- reproducing or studying the paper’s illustration logic,
- understanding model mechanics,
- building a cleaner package version later.

## Sensitivity Analysis

The second paper emphasizes sensitivity analysis and sub-model evaluation. The attached code reflects that design.

### Thermoregulation sensitivity
The thermoregulation script includes sensitivity-analysis style treatment of:

- wind speed,
- relative humidity,
- solar radiation,
- cloud cover,
- rainfall,
- total body weight,
- heat production.

The paper reports that lower and upper critical temperatures are especially sensitive to body temperature, latent heat release parameters, ambient temperature, and wind speed.

### Feed digestion sensitivity
The paper evaluates how feed constituents and digestion parameters affect:

- metabolisable energy content,
- digestible protein content.

The code structure supports exactly this constituent-based logic.

## Evaluation and Credibility

The attached papers report three layers of scientific credibility for LiGAPS-Beef:

1. **Model description and illustration** for conceptual transparency.
2. **Sensitivity analysis and sub-model evaluation** for thermoregulation and feed digestion.
3. **Whole-model evaluation** using independent animal-level data from multiple countries.

The third paper reports that the whole model was evaluated with independent data from Australia, Uruguay, and the Netherlands, and that the model could identify daily defining and limiting factors with acceptable accuracy for its intended use.

## How to Use This Code

Because the attached files are research-style scripts rather than a packaged library, the practical workflow is simple:

1. Place all required files in the same working directory.
2. Ensure Python dependencies are installed, especially:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - optionally `tqdm`
3. Ensure the climate CSV file is available at the expected path.
4. Run the relevant script directly with Python.

Example:

```bash
python LiGAPSBeef20180301_herd_worked.py
```

For standalone exploration:

```bash
python Thermoregulation_submodel_20170922_worked.py
python Feed_digestion_submodel20170907_worked.py
```

## Software Notes

### Nature of the implementation
This repository is a Python translation of an R-based scientific workflow. The conversion preserves logic as closely as possible, but users should keep in mind:

- the original published implementation was written in R,
- some arrays and loops are translated in an R-like way,
- some code remains script-oriented rather than modular,
- some comments still refer directly to R behavior or line mappings.

### Practical additions in the Python version
The integrated Python script includes practical enhancements such as:

- `Path`-based path handling,
- `matplotlib` non-interactive backend selection,
- optional `tqdm` progress display,
- debug toggles,
- explicit NumPy array management.

These additions support execution but are not part of the original scientific formalism.

## Limitations

### Scientific limitations
- The model uses a **daily** time step and therefore does not simulate within-day behavioral adaptation in detail.
- The thermoregulation model is calibrated for average cattle behavior and not minute-by-minute dynamics.
- Whole-model realism still depends on the quality of climate, feed, and breed inputs.

### Software limitations
- The code is still **research-script style**, not a clean package.
- Some sections remain tightly coupled and long.
- Standalone sub-model files and the integrated herd workflow are not yet organized as reusable Python modules.
- The current herd script appears tailored to selected illustration cases rather than a generalized command-line interface.

## Conclusions

The attached Python code set is a scientifically grounded implementation of the LiGAPS-Beef framework. It captures the core contribution of the papers:

- simulation of potential and feed-limited beef production,
- identification of defining and limiting factors,
- mechanistic integration of thermal balance, digestion, and growth,
- aggregation from animal-level to herd-level performance.

As a research codebase, it is valuable for:

- studying the LiGAPS-Beef structure,
- reproducing illustration logic,
- understanding thermoregulation and digestion assumptions,
- serving as a basis for future modularization, packaging, validation, and benchmarking.

The most important practical conclusion is that this repository is best treated as a **faithful translated research implementation** grounded in the source papers, and as a strong starting point for further engineering rather than as a finished software product.

## Suggested Next Steps

- modularize the integrated script into package components,
- separate constants, breed libraries, weather loading, plotting, and reporting,
- add regression tests against known outputs,
- formalize CLI entry points,
- document input data formats more explicitly,
- verify numerical equivalence with the original R outputs where possible.

## References

### Primary LiGAPS-Beef papers

1. Van der Linden, A., van de Ven, G.W.J., Oosting, S.J., van Ittersum, M.K., and de Boer, I.J.M. (2019). *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 1: model description and illustration*. Animal, 13(4), 845–855.
2. Van der Linden, A., van de Ven, G.W.J., Oosting, S.J., van Ittersum, M.K., and de Boer, I.J.M. (2019). *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 2: sensitivity analysis and evaluation of sub-models*. Animal, 13(4), 856–867.
3. Van der Linden, A., van de Ven, G.W.J., Oosting, S.J., van Ittersum, M.K., and de Boer, I.J.M. (2019). *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 3: model evaluation*. Animal.

### Core methodological references cited by the source papers

4. Chilibroste, P., Tamminga, S., and Boer, H. (1997). Rumen digestion and passage formalism used as the basis for feed digestion in LiGAPS-Beef.
5. Jarrige, R. et al. (1986). INRA fill-unit system used to constrain feed intake.
6. NRC (2000). Energy interpretation used for cattle and conversion concepts used in the LiGAPS-Beef framework.
7. McGovern, R.E. and Bruce, J.M. (2000). Thermoregulation modeling basis.
8. Turnpenny, J.R. et al. (2000). Thermoregulation modeling basis.
9. Van de Ven, G.W.J. et al. (2003). Production ecology concepts underlying yield-gap analysis.
10. Van der Linden, A. et al. (2015). Prior production-ecology framing for livestock yield-gap analysis.

## Citation Guidance

If you use this repository in research, cite the three LiGAPS-Beef papers above and clearly state that the code is a Python translation of the published LiGAPS-Beef workflow.