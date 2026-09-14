// Optional local lessons. Static copies remain playable and explicitly say they are frozen.
window.localLessons = async function (game, apply) {
  const status = document.createElement('p');
  status.textContent = 'Frozen demo. Use python serve.py ' + game + ' --learn for persistent local learning.';
  document.querySelector('header').appendChild(status);
  let enabled = false, latestGame = -1;
  function update(data) { if (data.games >= latestGame) { latestGame = data.games; apply(data.net); } }
  try {
    const response = await fetch('/learning/' + game);
    if (response.ok) {
      const data = await response.json(); enabled = data.enabled;
      if (enabled) { update(data); status.textContent = `Learning locally · ${data.games} completed lessons`; }
    }
  } catch (_) {}
  return async episode => {
    if (!enabled || !episode.drives.length) return;
    status.textContent = 'Reviewing the game and revisiting teacher examples…';
    try {
      const response = await fetch('/learning/' + game, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(episode)});
      if (!response.ok) throw Error(await response.text());
      const data = await response.json();
      if (data.games < latestGame) return;
      update(data);
      status.textContent = `Lesson ${data.games} saved · ${data.lesson.retained ? 'update retained' : 'earlier skills retained; episode saved for review'}`;
    } catch (error) { status.textContent = 'Lesson could not be saved: ' + error.message; }
  };
};
