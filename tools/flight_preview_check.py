"""Verify the actual full recording through a Pages-style static subdirectory.

Runs in repository browser CI, or against the public URL for deployment checks.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, output):
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url+'recording/manifest.json') as response:
        manifest = json.load(response)
    assert len(manifest['frames']) == 528
    assert len(manifest['events']) == 2637
    assert sum(e.get('steps', 0) for e in manifest['events']) == 10870
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('window.replayState?.ready', timeout=180000)
        assert page.locator('meta[name="robots"]').get_attribute('content') == 'noindex, nofollow'
        state = page.evaluate('window.replayState')
        assert state['neurons'] == 38577 and state['synapses'] == 1056874
        # one drawn vertex per stored route address, one drawn synapse per stored value
        assert state['totalVertices'] == 38577+6308+2
        assert state['totalEdges'] == 1056874+6308*28+512
        page.locator('#step').click()
        page.wait_for_function('window.replayState.step===1', timeout=60000)
        assert page.locator('#screen').get_attribute('src').endswith('000000.png')
        page.screenshot(path=str(output/'flight-repairs.png'))
        page.locator('#clock').select_option('game')
        page.locator('#mode').select_option('activity')
        # Seek rather than walking there: this exercises independently loaded
        # base snapshots, including later parameter and route-memory versions.
        positive = next(e for e in manifest['events'] if e['kind'] == 'plasticity' and e['learning']['dopamine'] > 0 and e['frame'] > 30)
        for frame in [positive['frame'], 264, 527]:
            page.locator('#timeline').fill(str(frame))
            page.locator('#timeline').dispatch_event('input')
            page.wait_for_function('(n)=>window.replayState.frame===n', arg=frame, timeout=180000)
            expected = manifest['frames'][frame]
            assert page.locator('#screen').get_attribute('src').endswith(expected['image'])
            assert page.locator('#points').inner_text() == f"{expected['points']:,}"
            feedback = next(manifest['events'][i] for i in expected['events'] if manifest['events'][i]['kind'] == 'plasticity')
            assert abs(float(page.locator('#dopamine').inner_text())-feedback['learning']['dopamine']) < 0.000006
            assert page.evaluate('window.replayState.criticBias') == feedback['critic_bias']
            assert page.evaluate('window.replayState.waveformSteps') > 0
            page.screenshot(path=str(output/f'flight-decision-{frame}.png'))
        assert page.evaluate('window.replayState.kind') == 'end'
        assert float(page.locator('#dopamine').inner_text()) < 0
        # Full graph remains alongside the game on desktop and fits on mobile.
        game, brain = page.locator('.game').bounding_box(), page.locator('.brain').bounding_box()
        assert abs(game['y']-brain['y']) < 2 and brain['x'] > game['x']
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(750)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(output/'flight-mobile.png'), full_page=True)
        # The optional movie remains a separate, lightweight gameplay-only view.
        page.locator('details').first.locator('summary').click()
        page.locator('video').evaluate('(v)=>v.load()')
        page.wait_for_function('document.querySelector("video").readyState>=1', timeout=60000)
        duration = page.locator('video').evaluate('(v)=>v.duration')
        assert 35 <= duration < 36
        assert not errors, errors
        browser.close()
    print('Actual Pages replay verified: full graph, repairs, random seeks, reward, memory bases, final loss, mobile and movie.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url')
    parser.add_argument('--site', type=Path, default=Path('runs/pages'))
    parser.add_argument('--output', type=Path, default=Path('runs/flight-browser'))
    args = parser.parse_args()
    server = None
    try:
        if not args.url:
            server = subprocess.Popen([sys.executable, '-m', 'http.server', '8766', '--bind', '127.0.0.1', '--directory', str(args.site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
        check(args.url or 'http://127.0.0.1:8766/previews/1943/', args.output)
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
