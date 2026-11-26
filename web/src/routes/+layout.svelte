<script lang="ts">
  import { onMount } from 'svelte';
  import '../lib/ui/fonts.css';
  import '../lib/ui/tokens.css';

  let { children } = $props();

  onMount(() => {
    if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
    void navigator.serviceWorker.register(`${import.meta.env.BASE_URL}service-worker.js`);
  });
</script>

{@render children()}
