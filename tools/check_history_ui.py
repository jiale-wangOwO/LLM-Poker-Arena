"""Verify the durable, collapsible hand narrative in the real browser.

Run against an isolated server, for example --base http://127.0.0.1:8083.
Only offline fixture seats are used. No real session or provider is modified.
The fixture session is stopped in finally, including after a failed assertion.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from cdp import Browser


CHECK = r"""(async()=>{
  const checks=[];
  const check=(value,message)=>{if(!value)throw new Error(message);checks.push(message);};
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const until=async(fn)=>{for(let i=0;i<150;i++){if(fn())return;await wait(100);}throw new Error('Timeout: '+fn.toString()+'; '+JSON.stringify({hand:snap?.hand_number,awaiting:snap?.awaiting_human,holding:snap?.holding_result,error:snap?.error}));};
  const request=async(path,body)=>{
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'API failure');return data;
  };
  const group=hand=>document.querySelector(`.hand-log[data-hand="${hand}"]`);
  const created=await request('/api/game',{
    seats:[{seat:0,kind:'human',name:'Fixture player'},{seat:3,kind:'ai',persona:'rock'}],
    offline:true,starting_chips:1000,speed_seconds:0,seed:3,max_hands:3,reveal_all:false,
  });
  window.historyFixtureSession=created.session.session;
  window.historyFixtureSessions=[created.session.session];
  sessionId=created.session.session;snap=mergeReplay(created.session);startPolling();
  await until(()=>snap.awaiting_human&&group(1));
  const first=snap.hands.find(hand=>hand.hand_number===1);
  const forced=first.entries.filter(entry=>entry.kind==='blind');
  check(forced.length===2&&forced.some(entry=>entry.role==='small')&&forced.some(entry=>entry.role==='big'),'Both posted blinds exist before the first voluntary action');
  check(group(1).querySelectorAll('.history-forced').length===2&&!group(1).querySelector('.lg'),'The first live narrative visibly starts with posted blinds');
  check(group(1).open,'The current hand starts expanded');
  await sendAction('fold');
  await until(()=>snap.holding_result&&group(1)?.querySelector('.history-result'));
  check(snap.hold_total===20&&group(1).open,'The completed result remains expanded during the 20-second hold');
  check(group(1).querySelector('.history-winners').textContent.includes(snap.hand_result.winners[0]),'The completed hand displays the final winner');
  check(group(1).querySelector('.history-result-title').textContent.includes(fmt(snap.hand_result.pot)),'The completed hand displays the awarded pot');
  check(group(1).querySelector('.history-net-label')&&group(1).querySelectorAll('.rp-net').length===2,'The result displays both players net chip changes');
  check(group(1).querySelector('.history-refund').textContent.includes('uncalled bet returned'),'The actual fold settlement distinguishes the uncalled blind refund');
  const firstResult=structuredClone(snap.hand_result);
  const heldActions=snap.actions.length;
  await poll();
  check(snap.actions.length===heldActions&&group(1).querySelector('.history-result'),'The final result survives polls with no new actions');
  $('btnSkipResult').click();
  await until(()=>snap.hand_number===2&&group(2));
  check(!group(1).open&&group(2).open,'The previous hand automatically collapses when the next hand starts');
  const summary=group(1).querySelector('summary');
  check(summary.textContent.includes(firstResult.winners[0])&&summary.textContent.includes('Complete')&&summary.textContent.includes(fmt(firstResult.pot)),'The collapsed header retains the winner and final pot');
  summary.click();await wait(50);
  check(group(1).open&&!followHistory,'A manually expanded old hand disables automatic following');
  await poll();
  check(group(1).open,'A manually expanded old hand survives polling');
  for(let i=0;i<80&&!snap.holding_result;i++){
    if(snap.awaiting_human)await sendAction(snap.legal.can_check?'check':'call');
    else {await wait(100);await poll();}
  }
  await until(()=>snap.holding_result&&snap.hand_result.hand===2);
  check(group(1).open&&group(2).querySelector('.history-result'),'New betting actions preserve an expanded previous hand and finish with a result');
  const second=snap.hands.find(hand=>hand.hand_number===2);
  const delta=await (await fetch(`/api/game/${sessionId}?history_after=1`)).json();
  check(delta.session.hands.length===1&&delta.session.hands[0].hand_number===2,'History polling can request only hands after its completed-hand cursor');
  snap=mergeReplay(delta.session);render();
  check(snap.hands.length===2&&group(1).querySelector('.history-result'),'Delta polling retains an earlier completed hand and its result in the browser cache');
  if(second.result.showdown.length){
    check(group(2).querySelectorAll('.history-reveal').length===second.result.showdown.length,'All publicly tabled cards and hand rankings are displayed');
  }
  const preflopAction=snap.actions.find(action=>action.hand===2&&action.replay&&!action.replay.hand_over&&action.replay.board.length===0);
  check(!!preflopAction,'The fixture includes a preflop replay before settlement');
  const replayRow=group(2).querySelector(`[data-history-seq="${preflopAction.seq}"]`);
  replayRow.focus();replayRow.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  check(viewSeq===preflopAction.seq&&viewIndex!==null,'The Enter key opens an action replay');
  check(!group(2).querySelector('.history-result')&&!group(2).querySelector('.history-board'),'Earlier replay hides the final result and future result board');
  check([...group(2).querySelectorAll('.history-street .card')].length===0&&stateAt(viewIndex).board.length===0,'Preflop replay narration and table hide future public cards');
  $('btnLive').click();
  check(viewIndex===null&&group(2).querySelector('.history-result'),'Returning to Live restores the completed result');
  const review=group(2).querySelector('[data-review-seq]');
  check(!!review,'The final result offers review of its final captured table');
  review.click();
  check(stateAt(viewIndex).hand_over&&group(2).querySelector('.history-result'),'Final-table review retains the settlement result');
  $('btnLive').click();
  $('btnCollapseHistory').click();
  check([...document.querySelectorAll('.hand-log')].every(node=>!node.open)&&!followHistory,'Collapse all closes every hand and pauses following');
  $('btnFollowHistory').click();
  check(!group(1).open&&group(2).open&&followHistory,'Follow current opens the current hand and folds previous hands');
  group(1).querySelector('summary').click();await wait(50);
  $('btnSkipResult').click();
  await until(()=>snap.hand_number===3&&group(3));
  check(group(1).open,'An explicitly expanded old hand survives a hand boundary');
  for(let i=0;i<80&&!snap.holding_result;i++){
    if(snap.awaiting_human)await sendAction('fold');
    else {await wait(100);await poll();}
  }
  await until(()=>snap.holding_result&&snap.hand_result.hand===3);
  check(group(3).querySelector('.history-result')&&$('btnSkipResult').textContent.includes('finish session'),'The final hand has a durable result during the last hold');
  $('btnSkipResult').click();
  await until(()=>snap.finished&&!snap.holding_result);
  check(group(3).querySelector('.history-result')&&snap.hands.length===3,'The final result persists after Skip finishes the capped session');

  // Controlled presentation fixtures cover uncommon split/side-pot results
  // and a very long narrative without changing the engine or any saved game.
  clearInterval(timer);timer=null;
  const real=structuredClone(snap),realView=viewIndex,realSeq=viewSeq;
  try{
    const hand=snap.hands.at(-1);
    hand.players=[{seat:0,name:'Split A'},{seat:3,name:'Split B'}];
    hand.result={...hand.result,pot:240,winners:['Split A','Split B'],board:['As','Ks','Qs','Js','Ts'],
      payouts:[{name:'Split A',amount:60,pot_label:'Main pot',split:true},{name:'Split B',amount:60,pot_label:'Main pot',split:true},{name:'Split A',amount:120,pot_label:'Side pot 1'},{name:'Split A',amount:25,reason:'uncalled_bet_returned'}],
      showdown:[{name:'Split A',cards:['2c','3c'],hand_name:'Royal flush'},{name:'Split B',cards:['4d','5d'],hand_name:'Royal flush'}],deltas:{0:0,3:0}};
    renderLog();
    check(group(3).querySelectorAll('.history-payout').length===3&&group(3).textContent.includes('Side pot 1')&&group(3).textContent.includes('(split)'),'Split main-pot and side-pot awards retain their own rows');
    check(group(3).querySelector('.history-refund').textContent.includes('25'),'Uncalled bets are separately displayed as returns');
    check(group(3).querySelectorAll('.history-reveal').length===2&&group(3).querySelectorAll('.rp-net').length===2,'Public reveals and zero net changes survive a split result');
    const sameCount=snap.actions.length;hand.result.pot=245;renderLog();
    check(snap.actions.length===sameCount&&group(3).querySelector('.history-result-title').textContent.includes('245'),'Result changes render even when action count is unchanged');
    hand.entries.push(...Array.from({length:90},(_,i)=>({kind:'action',seq:1000+i,name:'Fixture player',action:'check',street:'river',amount:0,pot:245,speech:'Long-hand scroll fixture '+i})));
    handOpenState.set(3,true);followHistory=false;renderLog();
    const box=$('log');box.scrollTop=180;const before=box.scrollTop;
    hand.entries.push({kind:'action',seq:1100,name:'Fixture player',action:'check',street:'river',amount:0,pot:245,speech:'New public action'});renderLog();
    check(box.scrollHeight>box.clientHeight*2&&Math.abs(box.scrollTop-before)<=1,'New actions do not pull a manually scrolled long hand to the bottom');
    $('btnFollowHistory').click();
    check(followHistory&&box.scrollHeight-box.scrollTop-box.clientHeight<3,'Follow current explicitly returns to the latest action/result');
  }finally{snap=real;viewIndex=realView;viewSeq=realSeq;handOpenState.clear();followHistory=true;render();}
  return {session:sessionId,checks:checks.length,passed:checks,hands:snap.hands.length,final_result:snap.hands.at(-1).result};
})()"""

KEYBOARD_FIXTURE = r"""(async()=>{
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const until=async(fn)=>{for(let i=0;i<100;i++){if(fn())return;await wait(100);}throw new Error('Keyboard fixture timed out');};
  const response=await fetch('/api/game',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    seats:[{seat:0,kind:'human',name:'Keyboard fixture'},{seat:3,kind:'ai',persona:'rock'}],
    offline:true,starting_chips:1000,speed_seconds:0,seed:3,max_hands:1,reveal_all:false,
  })});
  const created=await response.json();if(!response.ok)throw new Error(created.error);
  window.historyFixtureSession=created.session.session;
  window.historyFixtureSessions.push(created.session.session);
  sessionId=created.session.session;viewIndex=null;viewSeq=null;snap=mergeReplay(created.session);startPolling();
  await until(()=>snap.awaiting_human);
  await sendAction('fold');
  await until(()=>snap.holding_result&&document.querySelector('.hand-log[data-hand="1"] .history-result'));
  clearInterval(timer);timer=null;
  return sessionId;
})()"""


def wait_for_server(base: str) -> None:
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        try:
            with urllib.request.urlopen(base+'/',timeout=2):
                return
        except (urllib.error.URLError,TimeoutError):
            time.sleep(0.4)
    raise RuntimeError('Isolated history test server is not available: '+base)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:8083')
    args=parser.parse_args()
    wait_for_server(args.base)
    output=Path('logs');output.mkdir(exist_ok=True)
    browser=Browser(args.base+'/?session=history-check-empty',width=1600,height=1000,wait=1)
    try:
        report=browser.eval(CHECK,await_promise=True)
        # Native details keyboard activation must use a real browser key event.
        browser.eval("document.querySelector('.hand-log[data-hand=\"3\"] summary').focus()")
        was_open=browser.eval("document.querySelector('.hand-log[data-hand=\"3\"]').open")
        browser.send('Input.dispatchKeyEvent',type='keyDown',key='Enter',code='Enter',windowsVirtualKeyCode=13,text='\r',unmodifiedText='\r')
        browser.send('Input.dispatchKeyEvent',type='keyUp',key='Enter',code='Enter',windowsVirtualKeyCode=13)
        time.sleep(0.15)
        is_open=browser.eval("document.querySelector('.hand-log[data-hand=\"3\"]').open")
        assert is_open!=was_open,'Native keyboard activation did not toggle the details header: '+str(browser.eval("({focus:document.activeElement.outerHTML,wasOpen:"+json.dumps(was_open)+",isOpen:"+json.dumps(is_open)+",lobby:$('lobby').className})"))
        browser.eval('renderLog()')
        assert browser.eval("document.querySelector('.hand-log[data-hand=\"3\"]').open")==is_open,'Header keyboard state did not survive rendering'
        report['checks']+=1
        report['passed'].append('Native details keyboard activation persists across rendering')
        browser.eval("$('btnFollowHistory').click();document.activeElement.blur()")
        browser.screenshot(output/'ui-history.png')
        browser.send('Emulation.setDeviceMetricsOverride',width=390,height=844,deviceScaleFactor=1,mobile=False)
        fits=browser.eval("(()=>{const rail=$('rail').getBoundingClientRect();return {fits:document.documentElement.scrollWidth<=innerWidth,railLeft:rail.left,railRight:rail.right};})()")
        assert fits['fits'],f'Hand history exceeds the mobile width: {fits}'
        browser.eval("$('rail').scrollIntoView({block:'start'})")
        browser.screenshot(output/'ui-history-390.png')
        report['checks']+=1
        report['passed'].append('History and controls fit the 390px mobile viewport')
        browser.send('Emulation.setDeviceMetricsOverride',width=1600,height=1000,deviceScaleFactor=1,mobile=False)
        report['keyboard_session']=browser.eval(KEYBOARD_FIXTURE,await_promise=True)
        for key,code,vk,pressed_text in [('Enter','Enter',13,'\r'),(' ','Space',32,' ')]:
            browser.eval("document.querySelector('.hand-log[data-hand=\"1\"] summary').focus()")
            was_open=browser.eval("document.querySelector('.hand-log[data-hand=\"1\"]').open")
            browser.send('Input.dispatchKeyEvent',type='keyDown',key=key,code=code,windowsVirtualKeyCode=vk,text=pressed_text,unmodifiedText=pressed_text)
            browser.send('Input.dispatchKeyEvent',type='keyUp',key=key,code=code,windowsVirtualKeyCode=vk)
            time.sleep(0.15)
            toggled=browser.eval("document.querySelector('.hand-log[data-hand=\"1\"]').open")
            assert toggled!=was_open,f'{code} did not toggle the hand summary during the result hold'
            browser.eval('poll()',await_promise=True)
            assert browser.eval('snap.holding_result&&!snap.finished'),f'{code} on the hand summary unexpectedly skipped the 20-second result hold'
            assert browser.eval("document.querySelector('.hand-log[data-hand=\"1\"]').open")==toggled,f'{code} summary state did not persist through polling'
            report['checks']+=1
            report['passed'].append(f'Native {code} toggles a hand during the 20-second hold without skipping it')
        print(json.dumps(report,indent=2))
    finally:
        try:
            browser.eval("(async()=>{clearInterval(timer);for(const id of window.historyFixtureSessions||[])await fetch(`/api/game/${id}/control`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})});})()",await_promise=True)
        finally:
            browser.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
