"""Check independent, draggable player windows in a real headless browser.

Use a dedicated server: python tools/check_player_windows.py --base http://127.0.0.1:8084
Creates and stops only its own offline fixture; no provider settings are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdp import Browser


SETUP = r"""(async()=>{
  const checks=[];
  const check=(condition,message)=>{if(!condition)throw new Error(message);checks.push(message);};
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const until=async(fn)=>{for(let i=0;i<150;i++){if(fn())return;await wait(100);}throw new Error('Timeout: '+fn.toString());};
  const request=async(path,body)=>{
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'API failure');return data;
  };
  const card=seat=>document.querySelector(`.pcard[data-seat="${seat}"]`);
  const ready=seat=>card(seat)&&!card(seat).querySelector('.pc-body').textContent.startsWith('Loading');
  const clickSeat=seat=>document.querySelector(`#felt .seat[data-seat="${seat}"]`).click();
  const created=await request('/api/game',{
    seats:[{seat:0,kind:'human',name:'Window fixture'},
      {seat:2,kind:'ai',persona:'rock'},{seat:4,kind:'ai',persona:'pro'}],
    offline:true,starting_chips:1000,speed_seconds:0,seed:42,max_hands:2,reveal_all:false,
  });
  window.playerWindowFixtureSession=created.session.session;
  sessionId=created.session.session;snap=mergeReplay(created.session);startPolling();
  await until(()=>snap.awaiting_human);
  await request(`/api/game/${sessionId}/control`,{action:'pause'});
  await request(`/api/game/${sessionId}/action`,{action:snap.legal.can_check?'check':'call',turn_token:snap.turn_token});
  await until(()=>snap.actions.some(action=>action.seat===0));
  await request(`/api/game/${sessionId}/control`,{action:'step',count:1});
  await until(()=>snap.actions.some(action=>action.seat===2||action.seat===4));
  check(snap.paused&&!snap.holding_result,'Fixture stops at a stable live hand after an AI decision');
  const ai=snap.players.filter(player=>player.seated&&player.is_ai);
  const primary=ai.find(player=>snap.actions.some(action=>action.seat===player.seat)).seat;
  const other=ai.find(player=>player.seat!==primary).seat;
  window.playerWindowCheck={checks,check,until,request,card,ready,clickSeat,primary,other};
  check(!$('focusPanel')&&!$('playerDock'),'The right column contains hand history without focus or inspector docks');
  check($('rail').querySelector('#log')&&$('rail').querySelector('#scrub'),'The right column retains history and replay controls');
  clickSeat(primary);await until(()=>ready(primary));
  clickSeat(other);await until(()=>ready(other));
  check(document.querySelectorAll('.pcard').length===2,'Clicking two players opens two independent windows');
  check([primary,other].every(seat=>card(seat).parentElement===document.body&&getComputedStyle(card(seat)).position==='fixed'),'Both player windows float above the page');
  check([primary,other].every(seat=>!card(seat).querySelector('.pc-hand')&&!card(seat).querySelector('.rec .w')),'Normal view hides opponent cards and private decision notes in both windows');
  card(primary).querySelector('[data-tab="profile"]').click();
  await until(()=>card(primary).dataset.tab==='profile'&&card(primary).querySelector('.bar'));
  check(card(primary).querySelector('.pc-body').textContent.includes('Voice')&&card(other).dataset.tab==='history','Each window keeps its own active tab');
  card(primary).querySelector('[data-tab="context"]').click();
  await until(()=>card(primary).dataset.tab==='context'&&!card(primary).querySelector('.ctx'));
  await loadPlayer(primary);
  check(!card(primary).querySelector('.ctx'),'Normal view does not expose the AI private context');
  const hidden=await (await fetch(`/api/game/${sessionId}/player/${primary}?context=1`)).json();
  check(!hidden.player.hole_cards&&hidden.player.context.length===0&&hidden.player.history.every(action=>!action.thought),'The player detail API also protects cards, context and reasoning');
  $('godToggle').click();
  await until(()=>snap.god_mode&&card(primary)?.querySelector('.ctx')&&card(other)?.querySelector('.pc-hand'));
  check(card(primary).querySelector('.ctx').textContent.length>50,'God mode refreshes a previously open private-context tab');
  clearInterval(timer);timer=null;
  await until(()=>!playerCards.get(primary).request);
  const normalFetch=window.fetch;
  let privateResponseHeld=false,releasePrivateResponse;
  const privateResponseGate=new Promise(resolve=>{releasePrivateResponse=resolve;});
  window.fetch=async(url,...args)=>{
    const response=await normalFetch(url,...args);
    if(String(url).endsWith(`/player/${primary}?context=1`)&&!privateResponseHeld){
      const body=await response.text();
      privateResponseHeld=true;await privateResponseGate;
      return new Response(body,{status:response.status,headers:response.headers});
    }
    return response;
  };
  const oldPrivateRequest=loadPlayer(primary);
  await until(()=>privateResponseHeld);
  $('godToggle').click();
  await until(()=>!snap.god_mode&&!card(primary)?.querySelector('.ctx')&&!card(other)?.querySelector('.pc-hand'));
  releasePrivateResponse();await oldPrivateRequest;window.fetch=normalFetch;startPolling();
  check(!card(primary).querySelector('.ctx'),'A delayed God-mode response cannot restore private context after God mode is off');
  $('godToggle').click();
  await until(()=>snap.god_mode&&card(primary)?.querySelector('.ctx'));
  card(primary).querySelector('[data-tab="history"]').click();
  await until(()=>card(primary).querySelector('.pc-hand')&&card(primary).querySelector('.rec'));
  check(card(primary).querySelector('.pc-hand .card')&&card(other).querySelector('.pc-hand .card'),'God mode reveals opponent cards in every open history window');
  $('godToggle').click();
  await until(()=>!snap.god_mode&&[primary,other].every(seat=>!card(seat)?.querySelector('.pc-hand')));
  check(!card(primary).querySelector('.rec .w'),'Turning God mode off clears private decision notes without closing windows');
  card(primary).querySelector('[data-tab="context"]').click();await loadPlayer(primary);
  check(!card(primary).querySelector('.ctx'),'Turning God mode off clears private context when its tab is reopened');
  card(primary).querySelector('[data-tab="history"]').click();await loadPlayer(primary);
  const otherNode=card(other);
  clickSeat(primary);
  check(!card(primary)&&card(other)===otherNode,'Clicking the same seat closes only that player window');
  clickSeat(primary);await until(()=>ready(primary));
  card(primary).querySelector('.x').click();
  check(!card(primary)&&card(other)===otherNode,'The close button leaves the other window untouched');
  clickSeat(primary);await until(()=>ready(primary));
  openPlayer(primary);
  check(document.querySelectorAll('.pcard').length===2,'Bringing an existing player window forward does not create a duplicate');
  return {session:sessionId,primary,other,checks:checks.length};
})()"""


FINISH = r"""(async()=>{
  const {check,until,request,card,ready,clickSeat,primary,other}=window.playerWindowCheck;
  const first=card(primary),second=card(other);
  const firstBefore=first.getBoundingClientRect(),secondBefore=second.getBoundingClientRect();
  await poll();
  const firstAfter=first.getBoundingClientRect(),secondAfter=second.getBoundingClientRect();
  check(card(primary)===first&&card(other)===second&&firstBefore.left===firstAfter.left&&firstBefore.top===firstAfter.top&&secondBefore.left===secondAfter.left&&secondBefore.top===secondAfter.top,'Polling preserves both window elements and their dragged positions');
  document.activeElement.blur();
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
  check(!document.querySelector('.pcard'),'Escape closes player windows');
  clickSeat(primary);await until(()=>ready(primary));
  const replay=snap.actions.findIndex(action=>action.replay);
  check(replay>=0,'The fixture records an action frame for replay');
  viewIndex=replay;viewSeq=snap.actions[replay].seq;render();
  check(!document.querySelector('.pcard'),'Entering replay closes live player windows');
  openPlayer(primary);
  check(!document.querySelector('.pcard'),'Replay cannot reopen a live player inspector');
  $('btnLive').click();
  await request(`/api/game/${sessionId}/control`,{action:'resume'});
  for(let i=0;i<150&&!snap.holding_result;i++){
    if(snap.awaiting_human)await sendAction('fold');
    await new Promise(resolve=>setTimeout(resolve,100));await poll();
  }
  await until(()=>snap.holding_result&&document.querySelector('.hand-log[data-hand="1"] .history-result'));
  check(snap.hold_total===20&&$('resultCountdown').textContent.includes('s'),'Hand completion still displays the 20-second result countdown');
  const completed=document.querySelector('.hand-log[data-hand="1"]');
  check(completed.querySelectorAll('.history-forced').length===2&&completed.querySelector('.history-winners'),'Hand history retains both blinds and the final winner');
  check(completed.querySelector('.history-net-label')&&completed.querySelector('.history-payout'),'Hand history retains payouts and net chip changes');
  const hand=snap.hand_number;
  $('btnSkipResult').click();
  await until(()=>snap.hand_number>hand&&document.querySelector(`.hand-log[data-hand="${hand+1}"]`));
  check(!completed.open&&document.querySelector(`.hand-log[data-hand="${hand+1}"]`).open,'The completed hand collapses when the next hand starts');
  check(completed.querySelector('.history-result'),'Collapsed previous hands retain their result for expansion');
  return {session:sessionId,checks:window.playerWindowCheck.checks.length,passed:window.playerWindowCheck.checks};
})()"""


def window_rect(browser: Browser, seat: int) -> dict:
    return browser.eval(f"""(()=>{{
      const card=document.querySelector('.pcard[data-seat="{seat}"]');
      const box=card.getBoundingClientRect(),head=card.querySelector('.pc-head').getBoundingClientRect();
      return {{left:box.left,top:box.top,right:box.right,bottom:box.bottom,
        headX:head.left+80,headY:head.top+16,width:innerWidth,height:innerHeight}};
    }})()""")


def drag(browser: Browser, seat: int, x: float, y: float) -> tuple[dict, dict]:
    browser.eval(f"openPlayer({seat})")
    before = window_rect(browser, seat)
    browser.send('Input.dispatchMouseEvent', type='mousePressed', x=before['headX'],
                 y=before['headY'], button='left', buttons=1, clickCount=1)
    browser.send('Input.dispatchMouseEvent', type='mouseMoved', x=x, y=y,
                 button='left', buttons=1)
    browser.send('Input.dispatchMouseEvent', type='mouseReleased', x=x, y=y,
                 button='left', buttons=0, clickCount=1)
    return before, window_rect(browser, seat)


def assert_inside(box: dict) -> None:
    assert box['left'] >= 0 and box['top'] >= 0, box
    assert box['right'] <= box['width'] + 1 and box['bottom'] <= box['height'] + 1, box


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8084')
    parser.add_argument('--out', default='logs')
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(exist_ok=True)
    browser = Browser(args.base + '/?session=player-window-check-empty', width=1600, height=1000, wait=1)
    try:
        setup = browser.eval(SETUP, await_promise=True)
        primary, other = setup['primary'], setup['other']
        untouched = window_rect(browser, other)
        before, moved = drag(browser, primary, 120, 120)
        assert abs(moved['left'] - before['left']) > 10 or abs(moved['top'] - before['top']) > 10, (before, moved)
        assert_inside(moved)
        unchanged = window_rect(browser, other)
        assert untouched['left'] == unchanged['left'] and untouched['top'] == unchanged['top'], (untouched, unchanged)
        browser.eval("playerWindowCheck.check(true,'Dragging one player window does not move the other')")
        _, top_left = drag(browser, primary, 0, 0)
        assert_inside(top_left)
        _, bottom_right = drag(browser, primary, top_left['width'] - 1, top_left['height'] - 1)
        assert_inside(bottom_right)
        browser.eval("playerWindowCheck.check(true,'Real pointer dragging clamps the complete window to both viewport corners')")
        browser.screenshot(output / 'ui-player-windows.png')
        browser.send('Emulation.setDeviceMetricsOverride', width=390, height=844,
                     deviceScaleFactor=1, mobile=False)
        browser.eval("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))", await_promise=True)
        for seat in (primary, other):
            assert_inside(window_rect(browser, seat))
        assert not browser.eval('document.documentElement.scrollWidth>innerWidth'), 'Player windows cause horizontal overflow on mobile'
        browser.eval("playerWindowCheck.check(true,'Both windows remain reachable inside a 390px viewport after resizing')")
        browser.screenshot(output / 'ui-player-windows-390.png')
        browser.send('Emulation.setDeviceMetricsOverride', width=1600, height=1000,
                     deviceScaleFactor=1, mobile=False)
        result = browser.eval(FINISH, await_promise=True)
        print(json.dumps(result, indent=2))
    finally:
        try:
            browser.eval(r"""(async()=>{
              clearInterval(timer);
              if(window.playerWindowFixtureSession)await fetch(`/api/game/${window.playerWindowFixtureSession}/control`,{
                method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})
              });
            })()""", await_promise=True)
        finally:
            browser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
