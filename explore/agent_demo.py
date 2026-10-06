from shopwright.agent import run_agent

for q in ["I want Sony headphones under $100. Which one has the fewest complaints about comfort?",
          "Compare two good laptops under $500 for a college student.",
          "Do you sell refrigerators?"]:
    out = run_agent(q)
    print(f"\n=== {q}\nsteps={out['steps']}")
    for t in out["trace"]:
        print(f"  step {t['step']}: {t['tool']}({t['args']})  error={t['error']}  {t['ms']}ms")
    print("ANSWER:", out["answer"])
