"""Public reasoning and iteration policy enter through the real native ACP factory."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import build_hermes_environment, prepare_hermes_home


def test_native_acp_factory_public_parameters_reach_main_and_child_sdk(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes runtime is unavailable; this test does not invoke a model")
    config = replace(
        config,
        hermes_home=tmp_path / "hermes",
        hermes_api_key="private-test",
        hermes_base_url="http://127.0.0.1:8000/v1",
        resource_root=tmp_path / "resources",
    )
    public = tmp_path / "public.yaml"
    public.write_text(
        json.dumps(
            {
                "model": {"context_length": 262144, "max_tokens": 65536},
                "agent": {"reasoning_effort": "xhigh", "max_turns": 150},
                "providers": {
                    "omlx": {
                        "extra_body": {
                            "enable_thinking": True,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "chat_template_kwargs": {"preserve_thinking": True},
                        }
                    }
                },
            }
        )
    )
    home = config.hermes_home / "attempt"
    prepare_hermes_home(config, home, agent_tools=True, mcp={}, public_config_path=public)
    runtime = Path(__file__).resolve().parents[4] / "server/agent_runtime"
    script = r"""
import copy,importlib,importlib.util,json,os,socket,sys,types
from pathlib import Path
def deny_network(*args,**kwargs):raise AssertionError("Network is forbidden")
socket.socket.connect=deny_network
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(sys.argv[1])/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
bootstrap=load("bootstrap","hermes_acp_bootstrap.py");bootstrap.install_isolated_env_loader()
adapter=load("public_agent","hermes_public_agent.py")
from hermes_cli.config import load_config_readonly,resolve_turn_limit
from hermes_constants import parse_reasoning_effort
from agent.iteration_budget import IterationBudget
from hermes_state import SessionDB
from acp_adapter.session import SessionManager
import httpx
home=Path(os.environ["HERMES_HOME"]);config=load_config_readonly()
sent=[]
def respond(request):
    if request.method!="POST":return httpx.Response(404)
    body=json.loads(request.content);sent.append(body)
    if body.get("stream"):
        chunks=[{"id":"fake","object":"chat.completion.chunk","created":0,"model":body["model"],
            "choices":[{"index":0,"delta":{"role":"assistant","content":"deterministic response"},"finish_reason":None}]},
            {"id":"fake","object":"chat.completion.chunk","created":0,"model":body["model"],
            "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}]
        return httpx.Response(200,headers={"content-type":"text/event-stream"},
            content=("".join("data: "+json.dumps(row)+"\n\n" for row in chunks)+"data: [DONE]\n\n").encode())
    return httpx.Response(200,json={"id":"fake","object":"chat.completion","created":0,"model":body["model"],
        "choices":[{"index":0,"message":{"role":"assistant","content":"deterministic response"},"finish_reason":"stop"}]})
httpx.Client._transport_for_url=lambda self,url:httpx.MockTransport(respond)
database=SessionDB(db_path=home/"state.db")
manager=SessionManager(db=database)
# No AIAgent constructor replacement and no explicit reasoning/iteration kwargs.
agent=manager._make_agent(session_id="acp-public-main",cwd=str(home))
child=None
try:
    assert agent.reasoning_config is None
    assert agent.max_iterations==sys.maxsize
    assert agent.iteration_budget.max_total==sys.maxsize
    before={"max_tokens":agent.max_tokens,"request_overrides":copy.deepcopy(agent.request_overrides),
        "model":agent.model,"base_url":agent.base_url,"fallback":copy.deepcopy(agent._fallback_chain)}
    agent.client.chat.completions.create(**agent._build_api_kwargs([{"role":"user","content":"before ACP adaptation"}],[]))
    adapter.apply_public_agent_parameters(agent)
    assert agent.reasoning_config==parse_reasoning_effort("xhigh")
    assert agent.max_iterations==150 and agent.iteration_budget.max_total==150
    assert agent.iteration_budget.used==0
    assert agent._session_init_model_config["reasoning_config"]==parse_reasoning_effort("xhigh")
    assert agent._session_init_model_config["max_iterations"]==150
    assert before=={"max_tokens":agent.max_tokens,"request_overrides":agent.request_overrides,
        "model":agent.model,"base_url":agent.base_url,"fallback":agent._fallback_chain}
    agent.client.chat.completions.create(**agent._build_api_kwargs([{"role":"user","content":"after ACP adaptation"}],[]))
    native=importlib.import_module("tools.delegate_tool")
    child=native._build_child_preserving_parent_tools(task_index=0,goal="native child parameter contract",context=None,
        toolsets=None,model=agent.model,max_iterations=agent.max_iterations,task_count=1,parent_agent=agent,role="leaf")
    assert child.reasoning_config==agent.reasoning_config
    assert child.max_iterations==150 and child.iteration_budget.max_total==150
    assert child.iteration_budget is not agent.iteration_budget
    assert child.model==agent.model and child.base_url==agent.base_url and not child._fallback_chain
    child.client.chat.completions.create(**child._build_api_kwargs([{"role":"user","content":"child contract"}],[]))
    first,main,spawn=sent
    def effort(body):
        return body.get("reasoning_effort") or body.get("reasoning",{}).get("effort") or body.get("chat_template_kwargs",{}).get("reasoning_effort")
    assert effort(first) is None
    assert effort(main)=="xhigh",{key:main.get(key) for key in ("reasoning_effort","reasoning","chat_template_kwargs","enable_thinking")}
    assert effort(spawn)=="xhigh"
    for body in (first,main,spawn):
        assert body["model"]==agent.model
        assert body["enable_thinking"] is True
        assert body["chat_template_kwargs"]["preserve_thinking"] is True
        assert body["max_tokens"]==65536 and body["temperature"]==1.0 and body["top_p"]==0.95

    for raw,reasoning in [("37","high"),("none",False),(0,None),(None,"xhigh")]:
        fresh=types.SimpleNamespace(model=agent.model,max_iterations=9,reasoning_config=None,
            iteration_budget=IterationBudget(9),_session_init_model_config={"max_tokens":65536})
        cfg={"agent":{"max_turns":raw,"reasoning_effort":reasoning}}
        original=copy.deepcopy(cfg)
        adapter.apply_public_agent_parameters(fresh,cfg)
        assert fresh.max_iterations==resolve_turn_limit(raw)
        assert fresh.iteration_budget.max_total==resolve_turn_limit(raw)
        assert fresh.reasoning_config==parse_reasoning_effort(reasoning)
        assert fresh._session_init_model_config["max_tokens"]==65536 and cfg==original
    exhausted=types.SimpleNamespace(model=agent.model,max_iterations=9,reasoning_config=None,
        iteration_budget=IterationBudget(9),_session_init_model_config={"max_iterations":9})
    exhausted.iteration_budget.consume()
    try:adapter.apply_public_agent_parameters(exhausted,config)
    except RuntimeError:pass
    else:raise AssertionError("A live iteration counter cannot be reset")
    assert exhausted.iteration_budget.used==1 and exhausted.max_iterations==9 and exhausted.reasoning_config is None
    saved_env=os.environ.get("HERMES_MAX_ITERATIONS")
    try:
        os.environ["HERMES_MAX_ITERATIONS"]="19"
        for cfg,expected in [({"agent":{"max_turns":37},"max_turns":25},37),
            ({"agent":{"max_turns":None},"max_turns":25},25),({"agent":{}},19)]:
            fresh=types.SimpleNamespace(model=agent.model,iteration_budget=IterationBudget(9),_session_init_model_config={})
            adapter.apply_public_agent_parameters(fresh,cfg)
            assert fresh.max_iterations==expected and fresh.iteration_budget.max_total==expected
        override=types.SimpleNamespace(model=agent.model,iteration_budget=IterationBudget(9),_session_init_model_config={})
        adapter.apply_public_agent_parameters(override,{"agent":{"reasoning_effort":"low",
            "reasoning_overrides":{agent.model:"xhigh"}}})
        assert override.reasoning_config==parse_reasoning_effort("xhigh")
    finally:
        if saved_env is None:os.environ.pop("HERMES_MAX_ITERATIONS",None)
        else:os.environ["HERMES_MAX_ITERATIONS"]=saved_env
    unlimited_runs=[]
    for label,raw in (("null",None),("zero",0)):
        fresh=manager._make_agent(session_id="acp-unlimited-"+label,cwd=str(home))
        fresh.skip_background_review=True
        fresh.skip_context_files=True
        fresh.load_soul_identity=False
        fresh.save_trajectories=False
        try:
            local=copy.deepcopy(config);local["agent"]["max_turns"]=raw
            adapter.apply_public_agent_parameters(fresh,local)
            assert type(fresh.max_iterations) is int and fresh.max_iterations==sys.maxsize
            output=fresh.run_conversation(user_message="Return the deterministic test response.",task_id="unlimited-"+label)
            assert output["completed"] is True and output.get("failed") is not True
            assert output["final_response"]=="deterministic response"
            assert fresh.iteration_budget.used==1
            assert fresh.iteration_budget.remaining==sys.maxsize-1
            unlimited_runs.append(label)
        finally:fresh.close()
    # The actual ACP get_session/_restore path must preserve the original SDK
    # identity and transcript while applying current public parameters to its
    # newly constructed agent, even when old DB metadata had missing values.
    cold_path=home/"cold-state.db"
    seed=SessionDB(db_path=cold_path)
    seed.create_session(session_id="cold-original",source="acp",model=agent.model,model_config={
        "cwd":str(home),"provider":agent.provider,"base_url":agent.base_url,"api_mode":agent.api_mode,
        "reasoning_config":None,"max_iterations":sys.maxsize})
    original_history=[{"role":"user","content":"Keep the accepted artifact; do not submit it again."},
        {"role":"assistant","content":"Prior accepted artifact recorded."}]
    for message in original_history:seed.append_message("cold-original",**message)
    persisted_history=seed.get_messages_as_conversation("cold-original",repair_alternation=True)
    seed.close()
    class PublicSessions(SessionManager):
        made=0
        def _make_agent(self,**kwargs):
            created=super()._make_agent(**kwargs)
            adapter.apply_public_agent_parameters(created)
            created.skip_context_files=True;created.load_soul_identity=False
            created.skip_background_review=True;created.save_trajectories=False
            self.made+=1
            return created
    cold_database=SessionDB(db_path=cold_path)
    restored_manager=PublicSessions(db=cold_database)
    state=restored_manager.get_session("cold-original")
    assert state is not None and state.session_id=="cold-original"
    assert state.history==persisted_history
    assert state.cwd==str(home)
    assert state.agent is not agent and restored_manager.made==1
    restored=state.agent
    try:
        assert restored.reasoning_config==parse_reasoning_effort("xhigh")
        assert restored.max_iterations==150 and restored.iteration_budget.max_total==150
        assert restored.model==agent.model and restored.base_url==agent.base_url and not restored._fallback_chain
        assert cold_database.get_messages_as_conversation("cold-original")==persisted_history
        result=restored.run_conversation(user_message="Return a read-only deterministic continuation.",
            conversation_history=state.history,task_id="cold-restore-contract")
        assert result["completed"] is True and result.get("failed") is not True
        assert effort(sent[-1])=="xhigh" and sent[-1]["enable_thinking"] is True
        for message in original_history:
            assert any(item.get("role")==message["role"] and item.get("content")==message["content"] for item in sent[-1]["messages"])
        assert restored.iteration_budget.used==1 and restored.iteration_budget.remaining==149
        assert restored_manager.get_session("cold-original") is state
        assert restored_manager.update_cwd("cold-original",str(home)) is state
        assert restored_manager.made==1 and restored.iteration_budget.used==1
    finally:restored.close();cold_database.close()
    print(json.dumps({"native_acp_gap_reproduced":True,"main_effort":effort(main),"child_effort":effort(spawn),
        "max_iterations":agent.max_iterations,"budget_total":agent.iteration_budget.max_total,"normalization":True,
        "no_live_counter_reset":True,"other_parameters_unchanged":True,"unlimited_real_runs":unlimited_runs,
        "canonical_priority":True,"cold_resume_same_sid_history_and_public_wire":True,"hot_resume_no_budget_reset":True}))
finally:
    if child is not None:
        child.close()
        if child in agent._active_children:agent._active_children.remove(child)
    agent.close();database.close()
"""
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(runtime)],
        cwd=tmp_path,
        env=build_hermes_environment(config, home),
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["native_acp_gap_reproduced"]
    assert receipt["main_effort"] == "xhigh"
    assert receipt["child_effort"] == "xhigh"
    assert receipt["max_iterations"] == 150
    assert receipt["budget_total"] == 150
    assert receipt["normalization"]
    assert receipt["no_live_counter_reset"]
    assert receipt["other_parameters_unchanged"]
    assert receipt["unlimited_real_runs"] == ["null", "zero"]
    assert receipt["canonical_priority"]
    assert receipt["cold_resume_same_sid_history_and_public_wire"]
    assert receipt["hot_resume_no_budget_reset"]
