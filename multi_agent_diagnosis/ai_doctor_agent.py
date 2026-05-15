from typing import Annotated, Literal, cast
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.tool import ToolMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from tools import pubmed_tool
from fpdf import FPDF
from markdown2 import markdown
from local_model import chat_model as base_llm

llm = base_llm.bind_tools([pubmed_tool])


def generate_pdf(report_text: str) -> bytes:
    html = markdown(report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.write_html(html)
    return bytes(pdf.output())


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    name: str | None
    age: int | None
    gender: str | None
    history: str | None
    diagnosed: bool | None
    confirmed: bool | None
    report: str | None
    pdf_bytes: bytes | None


class SpecialistClassifier(BaseModel):
    message_type: Literal["physician", "surgeon", "pharmacologist"] = Field(
        ...,
        description="Classify which specialist is best suited for the patient's message"
    )


_SCHEMA_TYPE_WORDS = {"string", "integer", "boolean", "number", "array", "object"}


def _extract_query(value) -> str | None:
    """Recursively unwrap whatever nesting the model puts around the query string."""
    if isinstance(value, str):
        s = value.strip()
        if not s or s in _SCHEMA_TYPE_WORDS:
            return None
        if s.startswith("{"):
            import ast
            try:
                return _extract_query(ast.literal_eval(s))
            except Exception:
                pass
        return s
    if isinstance(value, dict):
        for key in ("query", "object", "value", "text", "input"):
            if key in value:
                result = _extract_query(value[key])
                if result:
                    return result
    return None


def _invoke_pubmed(tc: dict) -> ToolMessage | None:
    """Return a ToolMessage for a PubMed tool call, or None if the args are malformed."""
    query = _extract_query(tc["args"].get("query"))
    if not query:
        print(f"[PubMed] Skipped malformed tool call — args: {tc['args']}")
        return None
    print(f"[PubMed] Querying: {query!r}")
    result = pubmed_tool.invoke(query)
    print(f"[PubMed] Result preview: {result[:200]!r}")
    return ToolMessage(content=result, tool_call_id=tc["id"], name=pubmed_tool.name, additional_kwargs={"query": query})


def classify_specialist(state: State):
    history = "\n".join(
        f"{getattr(msg, 'name', None) or msg.type}: {msg.content}"
        for msg in state["messages"][-8:]
    )

    system = (
        "You are deciding which doctor should continue the conversation.\n"
        "Possible outputs (one keyword only):\n"
        "- cardiologist\n"
        "- neurologist\n"
        "- general practitioner\n\n"
        "Examples:\n"
        'User: "Sharp chest pain when climbing stairs."\n'
        "Output: cardiologist\n\n"
        'User: "Tingling in fingers, memory trouble."\n'
        "Output: neurologist\n\n"
        'User: "Fever and fatigue for two days."\n'
        "Output: general practitioner\n\n"
        "Now classify the following message:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": history},
    ]

    structured_llm = base_llm.with_structured_output(SpecialistClassifier)
    response = cast(SpecialistClassifier, structured_llm.invoke(messages))

    print("Classified message type:", response.message_type)
    return {"message_type": response.message_type}


def general_practitioner(state: State):
    patient_details = (
        f"Patient Details:\nName: {state.get('name', 'Unknown')}\n"
        f"Age: {state.get('age', 'Unknown')}\n"
        f"Gender: {state.get('gender', 'Unknown')}\n"
        f"History: {state.get('history', 'Unknown')}\n\n"
    )

    system_content = (
        "You are a caring and detail-oriented General Practitioner. First decide whether you truly "
        "need PubMed based on your internal knowledge. If yes, explain why and call PubMed. If not, proceed with your "
        "diagnosis without external search. Your role is to conduct initial consultations, collect comprehensive patient "
        "information, and identify common illnesses. Ask follow-up questions to gather a full picture of the patient's "
        "condition, including lifestyle factors. If symptoms are outside your scope, refer the patient to the appropriate "
        "specialist. Always prioritize patient comfort, clarity, and accuracy. Do not hallucinate any details, "
        "and use only the information that the patient gives you."
        + patient_details
    )

    gp_messages = [{"role": "system", "content": system_content}] + state["messages"]
    reply = llm.invoke(gp_messages)

    tool_calls = getattr(reply, "tool_calls", []) or []
    tool_messages: list = [msg for tc in tool_calls if (msg := _invoke_pubmed(tc))]
    if tool_messages:
        print(f"Tool calls detected: {[tc['name'] for tc in tool_calls]}")
        gp_messages = [{"role": "system", "content": system_content}] + state["messages"] + [reply] + tool_messages
        reply = llm.invoke(gp_messages)
    elif not isinstance(reply.content, str) or not reply.content.strip():
        reply = base_llm.invoke(gp_messages)

    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return {"messages": tool_messages + [{"role": "assistant", "content": text, "name": "General Practitioner"}]}


def cardiologist(state: State):
    system_content = (
        "You are a knowledgeable and experienced Cardiologist. You evaluate symptoms "
        "related to the heart and circulatory system such as chest pain, palpitations, shortness of breath, "
        "and dizziness. Ask targeted follow-up questions about cardiovascular history, medications, and risk factors. "
        "Provide a reasoned diagnosis or suggest relevant cardiac tests when needed."
    )

    cardio_messages = [{"role": "system", "content": system_content}] + state["messages"]
    reply = llm.invoke(cardio_messages)

    tool_calls = getattr(reply, "tool_calls", []) or []
    tool_messages: list = [msg for tc in tool_calls if (msg := _invoke_pubmed(tc))]
    if tool_messages:
        cardio_messages = [{"role": "system", "content": system_content}] + state["messages"] + [reply] + tool_messages
        reply = llm.invoke(cardio_messages)
    elif not isinstance(reply.content, str) or not reply.content.strip():
        reply = base_llm.invoke(cardio_messages)

    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return {"messages": tool_messages + [{"role": "assistant", "content": text, "name": "Cardiologist"}]}


def neurologist(state: State):
    system_content = (
        "You are a specialized and analytical Neurologist. Focus on symptoms related "
        "to the nervous system, including headaches, dizziness, numbness, memory loss, and coordination issues. Ask "
        "relevant neurological assessment questions to narrow down possible conditions. Consider medical history and "
        "recent symptom patterns to form a working diagnosis or recommend neurological evaluation."
    )

    neuro_messages = [{"role": "system", "content": system_content}] + state["messages"]
    reply = llm.invoke(neuro_messages)

    tool_calls = getattr(reply, "tool_calls", []) or []
    tool_messages: list = [msg for tc in tool_calls if (msg := _invoke_pubmed(tc))]
    if tool_messages:
        neuro_messages = [{"role": "system", "content": system_content}] + state["messages"] + [reply] + tool_messages
        reply = llm.invoke(neuro_messages)
    elif not isinstance(reply.content, str) or not reply.content.strip():
        reply = base_llm.invoke(neuro_messages)

    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return {"messages": tool_messages + [{"role": "assistant", "content": text, "name": "Neurologist"}]}


def check_diagnosis(state: State) -> dict:
    msg = state["messages"][-1].content.lower()
    diagnosis_messages = [
        {
            "role": "system",
            "content": (
                "Your goal is to determine if the doctor has reached a conclusive diagnosis. "
                "You must only reply using the following words: "
                "- true if a conclusive diagnosis has been reached, "
                "- false if a conclusive diagnosis has not been reached. "
                "A conclusive diagnosis is where a treatment/medicine is recommended, or where the patient is "
                "referred to for a future visit."
            ),
        },
        {"role": "user", "content": msg},
    ]

    response = llm.invoke(diagnosis_messages)
    diagnosed = "true" in response.content

    return {"diagnosed": diagnosed}


def reporter(state: State):
    reporter_messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical report author. Using only the patient data "
                "and final diagnosis previously gathered, create a clear, precise medical report with the following sections:\n\n"
                "---\n"
                "**Patient Information**\n"
                "- Name\n- Age\n- Gender\n- Medical History (brief summary)\n\n"
                "**Chief Complaint & Presenting Symptoms**\n\n"
                "**Physical Findings & Diagnostic Observations** (as described by doctor)\n\n"
                "**Final Diagnosis** (as determined by the physician agent)\n\n"
                "**Assessment & Rationale** (brief reasoning behind the diagnosis)\n\n"
                "**Treatment Recommendations or Next Steps** (medications, referrals, follow-up suggestions)\n\n"
                'End with a short courteous note: "Thank you for using our medical assistant service. Wishing you good health."\n\n'
                'Also, add this disclaimer in bold: "This is not an official report. The recommendations in this report '
                'are merely suggestions and must be taken with caution." '
                "--- Give only the report and don't add any additional text."
            ),
        },
        {"role": "user", "content": f"Here is the conversation from which you can summarize: {state}"},
    ]

    report = llm.invoke(reporter_messages)

    return {"messages": state["messages"], "diagnosed": state["diagnosed"], "report": report.content}


graph_builder = StateGraph(State)

graph_builder.add_node("classify_specialist", classify_specialist)
graph_builder.add_node("general_practitioner", general_practitioner)
graph_builder.add_node("cardiologist", cardiologist)
graph_builder.add_node("neurologist", neurologist)
graph_builder.add_node("check_diagnosis", check_diagnosis)
graph_builder.add_node("reporter", reporter)

graph_builder.add_edge(START, "classify_specialist")

graph_builder.add_conditional_edges(
    "classify_specialist",
    lambda s: s.get("message_type", "general practitioner"),
    {
        "cardiologist": "cardiologist",
        "neurologist": "neurologist",
        "general practitioner": "general_practitioner",
    },
)

graph_builder.add_edge("general_practitioner", "check_diagnosis")
graph_builder.add_edge("cardiologist", "check_diagnosis")
graph_builder.add_edge("neurologist", "check_diagnosis")

graph_builder.add_conditional_edges(
    "check_diagnosis",
    lambda s: "reporter" if s.get("diagnosed") else END,
    {"reporter": "reporter", END: END},
)

graph_builder.add_edge("reporter", END)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)


def run_chatbot():
    print("Hello! I'm your virtual medical assistant. Let me collect a few basic details to get started.")

    name = input("Name: ")
    age = int(input("Age: "))
    gender = input("Gender: ")
    history = input("Previous medical history: ")

    state = {
        "messages": [],
        "message_type": None,
        "name": name,
        "age": age,
        "gender": gender,
        "history": history,
        "diagnosed": False,
    }

    config = {"configurable": {"thread_id": "2"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "q":
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state, config)

        print(f"{state['messages'][-1].name}: {state['messages'][-1].content}")

        if state["diagnosed"]:
            print("Final report:")
            print(state["report"])
            break


if __name__ == "__main__":
    run_chatbot()
