<script lang="ts">
  import { onMount } from 'svelte';
  import StatusBar from '../../lib/ui/StatusBar.svelte';

  const steps = [
    {
      title: 'The line hides a mixture',
      copy: 'A diode sweep is fitted as a straight line. Its intercept is treated as the band gap, but the prefactor and ideality move it too.',
      href: '../',
      label: 'Open the landing scene',
    },
    {
      title: 'Start with the posterior',
      copy: 'The noisy line becomes 500 posterior curves on the report-anchored synthetic M0 case. This is a simulation, not a recovered measurement.',
      href: '../infer/',
      label: 'Open the infer view',
    },
    {
      title: 'Watch the evidence contract',
      copy: 'Nested sampling shrinks a live set while the evidence accumulates. The event stream is the scene; the number is produced in Rust.',
      href: '../evidence/',
      label: 'Open the evidence scenes',
    },
    {
      title: 'See what the data cannot pin down',
      copy: 'Fisher information turns the same model into a thin ellipsoid. The sloppy direction is the honest limit of the experiment.',
      href: '../evidence/#s4',
      label: 'Open Sloppiness',
    },
    {
      title: 'Ask for an instant answer',
      copy: 'A frozen local flow returns a posterior in milliseconds. SBC, TARP and OOD are gates; the page refuses to run without a passing report.',
      href: '../npe/',
      label: 'Open the instant posterior',
    },
    {
      title: 'Twin the sensor',
      copy: 'The plate, junction and sensor do not heat together. The first Phase 4 slice makes their lag explicit and recovers the sensor time constant from the synthetic series.',
      href: '../thermal/',
      label: 'Open the thermal twin',
    },
    {
      title: 'Read the ledger, then decide',
      copy: 'Evidence, LOO and prior sensitivity sit together. Disagreement is shown, not averaged away; the lost recordings remain unavailable.',
      href: '../ledger/',
      label: 'Open the evidence ledger',
    },
  ];

  let step = $state(0);
  let daylight = $state(false);
  const commit = typeof __COMMIT_HASH__ === 'string' ? __COMMIT_HASH__ : 'dev';
  const current = $derived(steps[step]);

  function move(delta: number) {
    step = Math.min(steps.length - 1, Math.max(0, step + delta));
  }

  onMount(() => {
    daylight = localStorage.getItem('bgo-theme') === 'daylight';
    document.documentElement.dataset.theme = daylight ? 'daylight' : 'darkroom';
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'ArrowRight') move(1);
      if (event.key === 'ArrowLeft') move(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
</script>

<svelte:head><title>Story — Band-Gap Observatory</title></svelte:head>

<div class="shell">
  <header class="top">
    <a href="../" class="mark" aria-label="Band-Gap Observatory home">Band-Gap Observatory</a>
    <a href="../">Exit story</a>
  </header>
  <main>
    <p class="eyebrow">Guided story · {step + 1} of {steps.length}</p>
    <section class="panel" aria-live="polite" aria-labelledby="story-title">
      <h1 id="story-title">{current.title}</h1>
      <p>{current.copy}</p>
      <a class="cta" href={current.href}>{current.label}</a>
      <div class="controls">
        <button type="button" onclick={() => move(-1)} disabled={step === 0}>Previous</button>
        <div class="dots" aria-hidden="true">
          {#each steps as _, index}
            <span class:active={index === step}></span>
          {/each}
        </div>
        <button type="button" onclick={() => move(1)} disabled={step === steps.length - 1}>
          Next
        </button>
      </div>
      <p class="hint">Use the left and right arrow keys to move through the story.</p>
    </section>
  </main>
  <StatusBar backend="static" core="story" phase="Phase 3" {commit} />
</div>

<style>
  .shell {
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
  }
  .top {
    width: var(--page);
    margin: 0 auto;
    padding: 1.2rem 0;
    border-bottom: var(--hairline);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }
  .top a {
    color: var(--ink-dim);
    text-decoration: none;
  }
  .mark {
    color: var(--ink) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  main {
    flex: 1;
    width: var(--page);
    margin: 0 auto;
    padding: clamp(2rem, 7vw, 5rem) 0;
  }
  .panel {
    padding: clamp(1.5rem, 5vw, 3rem);
    max-width: 780px;
    margin-top: 1rem;
  }
  h1 {
    font-size: var(--step-3);
    margin-bottom: 1rem;
    max-width: 18ch;
  }
  p {
    color: var(--ink-dim);
    max-width: 62ch;
  }
  .cta {
    display: inline-block;
    margin: 1rem 0 2rem;
  }
  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    border-top: var(--hairline);
    padding-top: 1rem;
  }
  button {
    min-height: 44px;
    padding: 0.6rem 1rem;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: transparent;
    color: var(--ink);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .dots {
    display: flex;
    gap: 0.4rem;
  }
  .dots span {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 50%;
    border: 1px solid var(--ink-dim);
  }
  .dots span.active {
    background: var(--accent);
    border-color: var(--accent);
  }
  .hint {
    font-size: 0.8rem;
    margin: 1rem 0 0;
  }
</style>
