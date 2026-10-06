"""Verify the 20-second result display, stable skip button and final-hand flow.

Uses passive fixture opponents; no external model requests or provider edits.
Run against an isolated server with --base http://127.0.0.1:8082.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdp import Browser

CHECK = r"""(async()=>{
  const wait = ms=>new Promise(r=>setTimeout(r,ms));
  const check=(v,m)=>{if(!v)throw new Error(m);};
  const until=async(fn)=>{for(let i=0;i<100;i++){if(fn())return;await wait(100);}throw new Error('Timeout: '+fn.toString());};
  const request=async(path,body)=>{
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await response.json();check(response.ok,data.error||'API failure');return data;
  };
  const created=await request('/api/game',{
    seats:[{seat:0,kind:'human'},{seat:3,kind:'ai',persona:'rock'}],
    offline:true,starting_chips:1000,speed_seconds:0,seed:3,max_hands:2,reveal_all:false,
  });
  sessionId=created.session.session;snap=mergeReplay(created.session);startPolling();
  await until(()=>snap.awaiting_human);
  await sendAction('fold');
  await until(()=>snap.holding_result&&$('btnSkipResult'));
  check(snap.hold_total===20,'Default duration is exactly20seconds for preflop folds');
  check($('resultPanel').classList.contains('on'),'Result card is visible');
  check($('resultCountdown').textContent.includes('s'),'Seconds countdown is readable');
  const hand=snap.hand_number,button=$('btnSkipResult'),countBefore=$('resultCountdown').textContent;
  button.focus();await wait(1300);await poll();
  check($('btnSkipResult')===button&&document.activeElement===button,'Polling preserves the skip button and keyboard focus');
  check(snap.hand_number===hand&&snap.holding_result,'No new hand is dealt while result is held');
  check($('resultCountdown').textContent!==countBefore,'Countdown decreases between polls');
  const nativeFetch=window.fetch;
  window.fetch=(url,...args)=>String(url).endsWith('/control')?Promise.reject(new Error('Test disconnect')):nativeFetch(url,...args);
  await skipHold();window.fetch=nativeFetch;
  check($('resultError').textContent&& !$('btnSkipResult').disabled,'Failed skip stays visible and can be retried');
  button.click();await until(()=>snap.hand_number>hand);
  check(snap.hand_number===hand+1,'Clicking skip advances one hand');
  const invalid=await request(`/api/game/${sessionId}/control`,{action:'skip-hold',hand});
  check(invalid.skipped===false,'Stale skip cannot preselect the next result');
  await until(()=>snap.awaiting_human||(snap.holding_result&&snap.hand_result.hand===2));
  if(snap.awaiting_human)await sendAction('fold');
  await until(()=>snap.holding_result&&snap.hand_result.hand===2&&$('btnSkipResult'));
  check(snap.hold_total===20&&!snap.finished,'Final hand also holds20seconds');
  check($('btnSkipResult').textContent.includes('finish session'),'Final-hand skip says it finishes the session');
  return {session:sessionId,checks:13,hold_total:snap.hold_total,final_hand:snap.hand_result.hand};
})()"""

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='http://127.0.0.1:8082')
    args=parser.parse_args()
    output=Path('logs')
    output.mkdir(exist_ok=True)
    browser=Browser(args.base+'/?session=result-check-empty',width=1600,height=1000,wait=1)
    try:
        report=browser.eval(CHECK,await_promise=True)
        browser.screenshot(output/'ui-result-20s.png')
        for width,height in [(320,844),(390,844),(1280,720)]:
            browser.send('Emulation.setDeviceMetricsOverride',width=width,height=height,deviceScaleFactor=1,mobile=False)
            fits=browser.eval("(()=>{const r=$('resultPanel').getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight;})()")
            assert fits,f'Result card exceeds viewport at{width}px'
            browser.screenshot(output/f'ui-result-{width}.png')
        finished=browser.eval(r"""(async()=>{
          $('btnSkipResult').click();
          for(let i=0;i<100;i++){if(snap.finished&&!snap.holding_result)return true;await new Promise(r=>setTimeout(r,100));}
          return false;
        })()""",await_promise=True)
        assert finished,'Final hand skip did not finish the session'
        report['final_skip']=finished
        print(json.dumps(report,indent=2))
    finally:
        browser.close()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
