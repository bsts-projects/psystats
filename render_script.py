import subprocess

problems = list(range(1,2))
documents = [
    "08 - Hypothesis_Test_z-Scores.qmd",
    "09 - Single-Sample_t-Tests.qmd",
    "10 - Independent-Samples_t-Tests.qmd",
    "11 - Dependent-Samples_t-Tests.qmd",
    "12 - one-way_ANOVA.qmd",
    "13 - repeated-measures_ANOVA.qmd",
    "14 - factorial_ANOVA.qmd"
]

tests = [
    "z-test",
    "one-sample t-test",
    "independent-samples t-test",
    "dependent-samples t-test",
    "one-way ANOVA",
    "repeated-measures ANOVA",
    "two-factor ANOVA"
]

for document, test in zip(documents, tests):
    for problem in problems:
        # Construct the command with the -P flag for parameters
        command = [
            "quarto", "render", f"{document}",
            "--output", f"{test}_{problem}.html"
        ]
        subprocess.run(command)