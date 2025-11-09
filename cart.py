import bppy as bp
from bppy.analysis.bprogram_converter import BProgramConverter
from bppy.model.sync_statement import *
from bppy.model.b_thread import *
import itertools

BE = lambda s: bp.BEvent(s)

class EvaluatorListener(bp.PrintBProgramRunnerListener):
	def starting(self, b_program):
		self.events = []
	def ended(self, b_program):
		pass
	def event_selected(self, b_program, event):
		print(event)
		self.events.append(event.name)
		if len(self.events) == 50:
			raise TimeoutError()

mode = bp.analysis_thread
items = {'A': 5, 'B': 4, 'C': 4.5, 'D': 2.5}

event_names = [BE('login_prompt'), BE('ignored'),BE('logged'), BE('force_login'), BE('added'), BE('skip')]
item_events = [BE(f'serve_{i}') for i in items.keys()]
event_list = event_names + item_events

@mode
def site(items):
    logged_in = False
    for item_name in items.keys():
        yield sync(request=BE(f'serve_{item_name}'))
        res = yield sync(waitFor=[BE(f'added'), BE(f'skip')])
        if not logged_in and res.name.endswith('added'):
            yield sync(request=BE('login_prompt'))
            res = yield sync(waitFor=[BE('logged'), BE('ignored')])
            if res == BE('logged'):
                logged_in = True
    if not logged_in:
        yield sync(request=BE('force_login'))
        yield sync(waitFor=BE('logged'))
    yield sync(block=event_list)

buy_func = lambda r: max(0, (r-2)/5)
@mode
def user(items):
    while True:
        q = yield sync(waitFor=event_list)
        if q.name.startswith('serve'):
            offer = q.name.split('_')[1]
            will_buy = buy_func(items[offer])
            decision = yield choice({'added':will_buy, 'skip':1-will_buy})
            yield sync(request=BE(decision))
        if q == BE('login_prompt'):
            decision = yield choice({'logged':0.4, 'ignored':0.6})
            yield sync(request=BE(decision))
        if q == BE('force_login'):
            yield sync(request=BE('logged'))
        else:
            yield sync(block=event_list)


       
prog_gen = lambda: bp.BProgram(bthreads=[site(items), user(items)],
            event_selection_strategy=bp.SimpleEventSelectionStrategy(),
            listener=EvaluatorListener())
prog = prog_gen()
#prog.run()


conv = BProgramConverter(prog_gen, event_names + item_events)
test = conv.to_prism('cart.prism')