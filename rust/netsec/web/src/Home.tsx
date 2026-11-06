export function Home({count}: {count?: number}) { return (
    <section id="home-view">
      <section className="page-heading" aria-labelledby="home-title">
        <div className="heading-copy"><div className="eyebrow"><span className="heading-line" aria-hidden="true"></span>LESS GUESSWORK. MORE CONTEXT.</div><h1 id="home-title">Your network.<br/><span>In focus.</span></h1><div className="hero-bottom"><p>A closer look at your traffic.<br/>Compare the signals. Understand the evidence.</p><a className="hero-scroll" href="#workspaces" aria-label="Explore NetSec workspaces">↓</a></div></div>
      </section>
      <div className="chapter-line"><span>INTRUSION ANALYSIS, WITH PERSPECTIVE.</span><span>01 / OBSERVE &nbsp; 02 / EXPLAIN &nbsp; 03 / UNDERSTAND</span></div>
      <section id="workspaces" className="home-workspaces" aria-labelledby="workspaces-title"><div className="workspace-title"><div><span className="eyebrow">ONE INPUT. MORE PERSPECTIVE.</span><h2 id="workspaces-title">Where do we begin?</h2></div><span id="home-model-count" className="workspace-caption">{count == null ? "Checking model availability…" : `${count} compatible models available · local inference`}</span></div><div className="route-cards">
        <a className="route-card" href="/csv"><div className="route-card-top"><span className="route-number">Your data</span><span aria-hidden="true">↗</span></div><h3>CSV analysis</h3><p>Choose several saved models or run extensive mode, then supply a CSV, an included sample or feature rows. Follow every result, disagreement and explanation.</p><div className="route-card-bottom"><span>Local saved-model inference</span><span>Open workspace →</span></div></a>
        <a className="route-card" href="/live"><div className="route-card-top"><span className="route-number">This PC</span><span aria-hidden="true">↗</span></div><h3>Live screening</h3><p>Start observations explicitly. Model availability follows the measurements this input can actually provide.</p><div className="route-card-bottom"><span>Compatible inputs only</span><span>Open workspace →</span></div></a>
      </div></section><section className="home-principle"><span className="eyebrow">EVIDENCE OVER ASSUMPTIONS.</span><p>Detection finds the signal. XAI explains a model’s decision.<br/>The local assistant puts measured evidence into English.</p><a href="/csv?input=sample">Explore a validation example ↗</a></section>
    </section>

); }
