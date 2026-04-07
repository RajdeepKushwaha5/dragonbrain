<script>
  export let text = '';
  export let type = 'info'; // 'info', 'key', 'wow'

  const config = {
    info: { icon: 'i', border: 'var(--accent)', bg: 'rgba(91,141,239,0.06)', color: 'var(--accent-bright)' },
    key:  { icon: '→', border: 'var(--gold)', bg: 'rgba(240,194,70,0.06)', color: '#e8d44d' },
    wow:  { icon: '!', border: 'var(--green)', bg: 'rgba(61,214,140,0.06)', color: '#6ee7a0' },
  };

  $: c = config[type] || config.info;
</script>

{#if text}
  <div class="insight" style="background: {c.bg}; border-left-color: {c.border}; color: {c.color};">
    <span class="icon" aria-hidden="true">{c.icon}</span>
    <span class="body">{@html text}</span>
  </div>
{/if}

<style>
  .insight {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.65rem 1rem;
    border-radius: var(--radius-sm);
    border-left: 3px solid;
    font-size: 0.84rem;
    line-height: 1.6;
    margin-top: 0.6rem;
  }

  .icon {
    flex-shrink: 0;
    font-size: 0.85rem;
    line-height: 1.4;
  }
</style>
