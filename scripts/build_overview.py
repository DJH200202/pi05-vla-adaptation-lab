"""Create a 60-second overview from existing clips only (Pillow + ffmpeg)."""
import argparse, json, subprocess, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--font',required=True)
a=p.parse_args();work=ROOT/'.cache/overview';work.mkdir(parents=True,exist_ok=True)
d=json.loads((ROOT/'demo/assets/data.json').read_text())
def font(n):return ImageFont.truetype(a.font,n)
def run(args):subprocess.run([a.ffmpeg,'-hide_banner','-loglevel','error','-y',*args],check=True)
def slide(name,chart,title,caption):
 im=Image.new('RGB',(1280,720),'#f5f7f8');draw=ImageDraw.Draw(im)
 draw.text((45,24),'PI 0.5 / VLA ADAPTATION LAB',font=font(18),fill='#128b89')
 draw.text((45,56),title,font=font(30),fill='#17283c')
 graph=Image.open(ROOT/f'demo/assets/charts/{chart}.png').convert('RGB');graph.thumbnail((1190,520))
 im.paste(graph,((1280-graph.width)//2,125))
 draw.text((45,675),caption,font=font(17),fill='#627186');path=work/(name+'.png');im.save(path)
 out=work/(name+'.mp4');run(['-loop','1','-i',str(path),'-t','6','-r','20','-c:v','libx264','-pix_fmt','yuv420p',str(out)]);return out
segments=[slide('intro','success-rates','Which modules should a pretrained VLA update?',
 '500 demonstrations | 3000 updates / group | 500 official initial states / model')]
titles=['Wider adaptation: B fails, C and D succeed','Counterexample: C succeeds, D fails',
        'Task 5 difference: D succeeds, C fails','A shared failure: neither C nor D succeeds']
for case,title in zip(d['cases'],titles):
 im=Image.new('RGB',(1280,720),'#f5f7f8');draw=ImageDraw.Draw(im)
 draw.text((35,24),'RECORDED CLOSED-LOOP EXECUTION / NOT LIVE INFERENCE',font=font(17),fill='#128b89')
 draw.text((35,64),title,font=font(31),fill='#17283c')
 draw.text((35,115),f'Task {case["task_id"]} / Initial state {case["initial_state_id"]} | matched initial state',font=font(20),fill='#627186')
 for j,g in enumerate('BCD'):
  rec=case['records'][g];txt=f'{g} | '+('SUCCESS' if rec['success'] else 'TIMEOUT')+f' | {rec["steps"]} steps'
  draw.text((j*426+18,185),txt,font=font(20),fill='#128b89' if rec['success'] else '#bd663e')
 for j,line in enumerate(textwrap.wrap(case['prompt'],width=92)):
  draw.text((35,550+30*j),line,font=font(22),fill='#17283c')
 draw.text((35,650),'Each panel: external camera + wrist camera. Ended clips hold their final frame.',font=font(18),fill='#627186')
 bg=work/(case['id']+'.png');im.save(bg);out=work/(case['id']+'.mp4')
 inputs=['-loop','1','-i',str(bg)]
 for g in 'BCD':inputs+=['-i',str(ROOT/f'demo/media/{case["id"]}/{g}.mp4')]
 filt=';'.join(f'[{j}:v]scale=426:252,tpad=stop_mode=clone:stop_duration=12,trim=duration=12,setpts=PTS-STARTPTS[v{j}]' for j in (1,2,3))
 filt+=';[v1][v2][v3]hstack=inputs=3[row];[0:v][row]overlay=1:240:shortest=1[out]'
 run([*inputs,'-filter_complex',filt,'-map','[out]','-t','12','-r','20','-c:v','libx264','-pix_fmt','yuv420p',str(out)])
 segments.append(out)
segments.append(slide('outro','parameter-tradeoff','C: 94.8% with 53.5% of D\'s trainable parameters',
 'C vs D: 14 C-only wins, 16 D-only wins. Parameter count is not memory or speed.'))
manifest=work/'concat.txt';manifest.write_text(''.join("file '"+str(x)+"'\n" for x in segments))
run(['-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(ROOT/'demo/overview.mp4')])
print('Created demo/overview.mp4 (60 seconds, derived from recorded clips; no model execution)')
