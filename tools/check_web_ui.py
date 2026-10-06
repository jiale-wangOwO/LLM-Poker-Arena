"""Browser regressions for the table workflow; uses offline seats as test fixtures.

Run against an isolated local server: python tools/check_web_ui.py --base http://127.0.0.1:8084
No provider configuration is changed and no external model is called by this check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdp import Browser

CHECK = r"""(async () => {
  const check = (value, message) => { if (!value) throw new Error(message); };
  const waitFor = async (fn) => {
    for (let i=0; i<120; i++) { if (fn()) return; await new Promise(r => setTimeout(r,100)); }
    throw new Error('Timed out: '+fn.toString()+'; lobby='+$('lobbyError').textContent+'; snap='+JSON.stringify({session:sessionId,awaiting:snap?.awaiting_human,actions:snap?.actions?.length,error:snap?.error}));
  };
  check(typeof openLobby === 'function', 'Page script did not initialize');
  $('btnNew').click();
  check($('lobby').classList.contains('on'), 'Start must review the lineup');
  $('btnFill').click(); $('btnJoin').click();
  check(seatedPlan().length === 6, 'Six-seat lineup');
  $('fOffline').checked = true;
  $('fBlinds').value = '20/10'; $('btnDeal').click();
  await waitFor(() => !!$('lobbyError').textContent);
  check($('lobby').classList.contains('on'), 'Start errors must stay visible');
  $('fBlinds').value = '10/20'; $('fSpeed').value = '0'; $('fHands').value = '3'; $('fSeed').value = '42';
  $('btnDeal').click();
  await waitFor(() => snap?.awaiting_human && $('actions').classList.contains('on'));
  check(!$('lobby').classList.contains('on'), 'Lobby closes after deal');
  check(document.querySelectorAll('.seat').length === 6, 'Every seat is on the table');
  const chosen = Number($('raiseSlider').min) + 1;
  setRaiseAmount(chosen);
  await poll();
  check(raiseAmount === chosen, 'Composed chip amount survives polling without rounding');
  check(!$('btnRaise').disabled && [...$('raiseQuick').querySelectorAll('button')].every(b=>!b.disabled), 'Raise controls are enabled');
  $('raiseAmount').focus(); $('raiseAmount').value='1'; $('raiseAmount').dispatchEvent(new Event('input'));
  await poll();
  check($('raiseAmount').value==='1','Partial focused amount survives polling');
  $('raiseAmount').blur(); setRaiseAmount(chosen);
  const allInSeat = snap.players.find(p => p.seated && p.is_ai).seat;
  const original = structuredClone(snap);
  snap.players[allInSeat].chips=0; snap.players[allInSeat].status='all_in'; snap.players[allInSeat].street_bet=1000;
  render();
  check(!!document.querySelector(`.seat[data-seat='${allInSeat}'] .wager.allin`), 'All-in seat stays visible with wager');
  check(!document.querySelector(`#outRail .outChip[data-seat='${allInSeat}']`), 'All-in is not eliminated');
  snap=original; render();
  openPlayer(allInSeat);
  await waitFor(() => document.querySelector(`.pcard[data-seat='${allInSeat}'] .pc-body`) && !document.querySelector(`.pcard[data-seat='${allInSeat}'] .pc-body`).textContent.includes('Loading'));
  const inspector = document.querySelector(`.pcard[data-seat='${allInSeat}']`);
  check(getComputedStyle(inspector).position === 'fixed' && inspector.parentElement === document.body, 'Player inspection floats independently of hand history');
  closePlayer(allInSeat);
  const transportFetch = window.fetch;
  window.fetch = (url,...args) => String(url).endsWith('/action') ? Promise.reject(new Error('Test disconnect')) : transportFetch(url,...args);
  await sendAction(snap.legal.can_check?'check':'call');
  window.fetch = transportFetch;
  check(snap.awaiting_human && !$('aCall').disabled, 'A failed POST can be retried on the same confirmed turn');
  const boundaries = [...document.querySelectorAll('#felt .seat, #actions')].filter(n=>n.getClientRects().length).map(n=> {
    const r=n.getBoundingClientRect();return {kind:n.className||n.id,left:r.left,right:r.right,top:r.top,bottom:r.bottom};
  });
  check(boundaries.every(r=>r.left>=0&&r.right<=innerWidth), 'Table controls fit the desktop width');
  let sent=0; const fetchOriginal=window.fetch;
  window.fetch=(url,...args)=> {if(String(url).endsWith('/action'))sent++;return fetchOriginal(url,...args);};
  const act=snap.legal.can_check?'check':'call';
  await Promise.all([sendAction(act),sendAction(act)]);
  window.fetch=fetchOriginal;
  check(sent===1, 'Double click submits exactly one move');
  await waitFor(()=>snap.actions.some(a=>a.replay));
  const index=snap.actions.findIndex(a=>a.replay);
  viewIndex=index; viewSeq=snap.actions[index].seq; render();
  const frame=stateAt(index);
  check(frame.hand_number===snap.actions[index].replay.hand_number&&JSON.stringify(frame.board)===JSON.stringify(snap.actions[index].replay.board),'Replay uses captured board');
  check(!$('actions').classList.contains('on'),'Replay has no actionable controls');
  openPlayer(allInSeat);
  check(!document.querySelector('.pcard'),'Replay cannot open a live inspector');
  $('btnLive').click();
  check(viewIndex===null&&viewSeq===null,'Return to live clears replay anchor');
  $('btnNew').click();
  $('btnLobbySettings').click();
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
  check(!$('settings').classList.contains('on')&&$('lobby').classList.contains('on'),'Escape closes only top dialog');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
  check(!$('lobby').classList.contains('on'),'Escape closes lineup');
  await waitFor(()=>snap.awaiting_human);
  const full = await (await fetch(`/api/game/${sessionId}`)).json();
  mergeReplay(full.session);
  const latest = Math.max(...replayCache.keys());
  const delta = await (await fetch(`/api/game/${sessionId}?replay_after=${latest}`)).json();
  check(!delta.session.actions.some(a=>a.replay),'Unchanged delta does not repeat frames');
  const merged = mergeReplay(delta.session);
  check(merged.actions.some(a=>a.replay),'Delta merge retains captured frames');
  const hidden = structuredClone(merged); hidden.god_mode=!merged.god_mode;
  hidden.actions.forEach(a=>delete a.replay);
  mergeReplay(hidden);
  check(replayCache.size===0,'Viewing-mode change clears cached private frames');
  snap=mergeReplay(full.session); render();
  return {session:sessionId,checks:22,actions:snap.actions.length,turn_token:!!snap.turn_token,controls:boundaries};
})()"""

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8084')
    parser.add_argument('--out', default='logs')
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(exist_ok=True)
    browser = Browser(args.base + '/?session=ui-check-empty', width=1600, height=1000, wait=1)
    try:
        result = browser.eval(CHECK, await_promise=True)
        browser.screenshot(output / 'ui-desktop.png')
        for width, height in [(320, 844), (390, 844), (768, 1024), (1280, 720)]:
            browser.send('Emulation.setDeviceMetricsOverride', width=width, height=height,
                         deviceScaleFactor=1, mobile=False)
            measured = browser.eval(r"""({width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth,
              controls:[...document.querySelectorAll('#actions button,#felt .seat')].filter(n=>n.getClientRects().length).map(n=>{
                const r=n.getBoundingClientRect();return {left:r.left,right:r.right};})})""")
            assert not measured['overflow'], measured
            assert all(n['left'] >= 0 and n['right'] <= width for n in measured['controls']), measured
            if width <= 540:
                overlaps = browser.eval(r"""(() => {
                  const seats=[...document.querySelectorAll('#felt .seat')].map(n=>n.getBoundingClientRect());
                  return seats.some((a,i)=>seats.slice(i+1).some(b=>Math.min(a.right,b.right)>Math.max(a.left,b.left)+1&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+1));
                })()""")
                assert not overlaps, f'Overlapping seats at {width}px'
            browser.screenshot(output / f'ui-{width}.png')
        print(json.dumps(result, indent=2))
        print('Desktop, 320/390px mobile, 768px tablet and 1280x720 controls fit.')
        stopped = browser.eval(r"""(async()=>{
          $('btnStop').click();
          for(let i=0;i<40;i++){if(snap.stopped&&snap.finished&&!snap.awaiting_human)return true;await new Promise(r=>setTimeout(r,100));}
          return false;
        })()""", await_promise=True)
        assert stopped, 'Stop did not settle the table'
        print('Stop settles the table and disables human actions.')
    finally:
        try:
            browser.eval(r"""(async()=>{
              clearInterval(timer);
              if(sessionId)await fetch(`/api/game/${sessionId}/control`,{
                method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})
              });
            })()""", await_promise=True)
        except Exception:
            pass
        browser.close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
