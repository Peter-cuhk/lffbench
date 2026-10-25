import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from PIL import Image

spec = importlib.util.spec_from_file_location('live_viewer', Path(__file__).parents[1]/'scripts/agent_viewer/live.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def append(self,name,row):
        with (self.root/name).open('a') as f:f.write(json.dumps(row)+'\n')

    def test_partial_append_and_truncation(self):
        p=self.root/'events.jsonl';p.write_bytes(b'{"ev":"call"')
        tail=live.Tail(p);self.assertEqual(tail.read(),[])
        with p.open('ab') as f:f.write(b'}\n')
        self.assertEqual(tail.read(),[{'ev':'call'}]);self.assertEqual(tail.read(),[])
        p.write_text('{}\n');self.assertEqual(tail.read(),[{}]);self.assertEqual(tail.generation,1)

    def test_robot_image_paths_are_not_yet_agent_observations(self):
        self.append('gateway_transcript.jsonl',{'ev':'tool','name':'robot','images':0,'output':['{"images":["frame.jpg"]}']})
        run=live.Run(self.root);run.update();self.assertEqual(run.frames(),[])
        self.append('gateway_transcript.jsonl',{'ev':'tool','name':'view_images','images':1,'is_error':True,'arguments':{'paths':['frame.jpg']}})
        run.update();self.assertEqual(run.frames(),[])

    def test_embedded_frame_matches_confirmed_observation(self):
        raw=io.BytesIO();Image.new('RGB',(4,4),(255,0,0)).save(raw,format='PNG');raw=raw.getvalue()
        self.append('gateway_transcript.jsonl',{'ev':'tool','name':'view_images','images':1,'output':['{"attached":["frame.png"]}'],'t':time.time()})
        self.append('codex_events.jsonl',{'type':'item.completed','item':{'type':'mcp_tool_call','tool':'view_images','result':{'content':[{'type':'text','text':'{"attached":["frame.png"]}'},{'type':'image','data':base64.b64encode(raw).decode()}]}}})
        run=live.Run(self.root);run.update();self.assertEqual(run.frames()[0][1],raw)
        self.assertIn('\033[38;2;255;0;0m',live.image_lines(raw,4,2,True)[0])
        # A repeated path with a new observation must not reuse old embedded bytes.
        self.append('gateway_transcript.jsonl',{'ev':'tool','name':'view_images','images':1,'output':['{"attached":["frame.png"]}'],'t':time.time()})
        run.update();self.assertIsNone(run.frames()[0][1])

    def test_overwritten_and_outside_files_not_claimed_as_input(self):
        p=self.root/'frame.jpg';p.write_bytes(b'not original');os.utime(p,(20,20))
        run=live.Run(self.root);run.view_paths=['frame.jpg','/etc/passwd'];run.view_at=10
        frames=run.frames();self.assertIsNone(frames[0][1]);self.assertIsNone(frames[1][1])
        self.assertIn('修改',frames[0][2])

    def test_batch_follows_next_active_run(self):
        p=self.root/'manifest.json'
        runs=[{'status':'running','log':'first'},{'status':'pending'}]
        p.write_text(json.dumps({'runs':runs}));self.assertEqual(live.choose_run(self.root)[0]['log'],'first')
        runs[0]['status']='done';runs[1]={'status':'running','log':'next'}
        p.write_text(json.dumps({'runs':runs}));self.assertEqual(live.choose_run(self.root)[0]['log'],'next')

    def test_native_image_retains_original_pixel_resolution(self):
        raw=io.BytesIO();Image.new('RGB',(512,512),(255,0,0)).save(raw,format='PNG');raw=raw.getvalue()
        sequence=live.native_image([raw,raw],80,20,'iterm')
        encoded=sequence.split(':',1)[1].split('\a',1)[0]
        image=Image.open(io.BytesIO(base64.b64decode(encoded)))
        self.assertEqual(image.size,(1024,512))
        self.assertEqual(image.getpixel((900,300)),(255,0,0))
        self.assertIn('preserveAspectRatio=1',sequence)

    def test_kitty_chunks_roundtrip_to_original_pixels(self):
        import re
        raw=io.BytesIO();Image.effect_noise((512,512),100).convert('RGB').save(raw,format='PNG');raw=raw.getvalue()
        sequence=live.native_image([raw],80,30,'kitty')
        chunks=re.findall(r'\x1b_G([^;]+);(.*?)\x1b\\',sequence)
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(data)<=4096 for _,data in chunks))
        self.assertTrue(chunks[-1][0].endswith('m=0'))
        image=Image.open(io.BytesIO(base64.b64decode(''.join(data for _,data in chunks))))
        self.assertEqual(image.size,(512,512))
        self.assertEqual(image.tobytes(),Image.open(io.BytesIO(raw)).tobytes())
        self.assertIn(f'i={live.KITTY_IMAGE_ID}',live.clear_native('kitty'))
        self.assertNotIn('d=A',live.clear_native('kitty'))

    def test_native_detection_conservative_and_redirect_stays_text(self):
        self.assertEqual(live.choose_renderer('auto',{'TERM_PROGRAM':'iTerm.app'}),'iterm')
        self.assertEqual(live.choose_renderer('auto',{'TERM':'xterm-kitty'}),'kitty')
        self.assertEqual(live.choose_renderer('auto',{'TERM_PROGRAM':'vscode'}),'ansi')
        self.assertEqual(live.choose_renderer('auto',{'TERM_PROGRAM':'iTerm.app','TMUX':'yes'}),'ansi')
        self.assertEqual(live.choose_renderer('iterm',{},tty=False),'ansi')

    def test_terminal_control_characters_are_removed(self):
        self.assertNotIn('\033',live.clean('\033[2Jinjected'))

    def test_claude_iterm_command_keeps_visible_preview(self):
        # Reproduce the user's existing command in the embedded terminal.
        Image.new('RGB',(32,32),(255,0,0)).save(self.root/'agentview.png')
        run=live.Run(self.root)
        run.view_paths=['agentview.png'];run.view_at=time.time()+1
        output=io.StringIO()
        with patch.dict(os.environ,{'TERM_PROGRAM':'claude-desktop'}), \
             patch.object(live,'Run',return_value=run), \
             patch.object(output,'isatty',return_value=True), \
             patch.object(live.sys,'stdout',output):
            live.main(['--run',str(self.root),'--once','--renderer','iterm',
                       '--camera','main','--width','100','--image-rows','10'])
        rendered=output.getvalue()
        self.assertIn('刷新 6s · ansi',rendered)
        self.assertIn('已切换为字符预览',rendered)
        self.assertIn('\033[38;2;255;0;0m',rendered)
        self.assertNotIn('\033]1337;',rendered)


if __name__=='__main__':unittest.main()
