# 自定义保险感知Agent实现
import logging
from typing import TypedDict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class InsuranceAwareAgentState(TypedDict):
    """保险感知Agent的状态模型"""
    messages: List[BaseMessage]
    user_query: str
    agent_scratchpad: List[dict] = []
    use_insurance_tool: bool = False


class InsuranceAwareReActAgent:
    """具有保险问题识别能力的ReAct Agent"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        detailed_logs: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.detailed_logs = detailed_logs
        self.graph = None
        # 创建工具字典以便快速查找
        self.tools_dict = {tool.name: tool for tool in tools}
        logger.info("InsuranceAwareReActAgent initialized")

    async def conditional_edge(self, state: InsuranceAwareAgentState):
        """条件边实现，判断用户问题是否与保险相关"""
        try:
            logger.debug("Starting Insurance-aware Conditional Edge")
            
            # 检查用户问题是否包含保险相关关键词
            user_query = state["user_query"].lower()
            insurance_keywords = [
                "保险", "车险", "健康险", "人寿险", "医疗险", 
                "平安保险", "保险理赔", "保险政策", "保单", 
                "投保", "保费", "保额", "保险条款"
            ]
            
            # 判断是否为保险相关问题
            is_insurance_related = any(keyword in user_query for keyword in insurance_keywords)
            
            if is_insurance_related:
                logger.info("Detected insurance-related query, routing to pingan_search tool")
                state["use_insurance_tool"] = True
                return "use_pingan_search"
            else:
                logger.info("Query is not insurance-related, routing to regular search tools")
                state["use_insurance_tool"] = False
                return "use_regular_search"
        except Exception as ex:
            logger.error("Error in conditional_edge: %s", ex)
            # 出错时默认使用常规搜索
            return "use_regular_search"

    async def agent_node(self, state: InsuranceAwareAgentState):
        """Agent节点，处理用户查询并生成响应"""
        try:
            logger.debug("Starting Agent Node")
            
            # 根据条件路由结果选择合适的工具
            if state["use_insurance_tool"]:
                # 使用平安保险搜索工具
                state["agent_scratchpad"].append({
                    "tool": "pingan_search",
                    "tool_input": {"query": state["user_query"]}
                })
            else:
                # 使用通用搜索工具
                state["agent_scratchpad"].append({
                    "tool": "internet_search",
                    "tool_input": {"query": state["user_query"]}
                })
            
            return state
        except Exception as ex:
            logger.error("Failed to call agent_node: %s", ex)
            raise

    async def tool_node(self, state: InsuranceAwareAgentState):
        """工具节点，执行工具调用"""
        try:
            logger.debug("Starting Tool Node")
            
            # 获取要调用的工具和输入
            agent_action = state["agent_scratchpad"][-1]
            tool_name = agent_action["tool"]
            tool_input = agent_action["tool_input"]
            
            # 执行工具调用
            if tool_name in self.tools_dict:
                logger.info(f"Calling tool: {tool_name}")
                tool_result = await self.tools_dict[tool_name].ainvoke(input=tool_input)
                
                # 将工具结果添加到状态中
                state["messages"].append(
                    HumanMessage(content=f"工具'{tool_name}'返回结果: {tool_result}")
                )
            else:
                logger.error(f"Tool '{tool_name}' not found")
                state["messages"].append(
                    HumanMessage(content=f"错误: 工具'{tool_name}'不存在")
                )
            
            return state
        except Exception as ex:
            logger.error("Failed to call tool_node: %s", ex)
            raise

    async def build_graph(self):
        """构建并编译Agent工作流图"""
        try:
            logger.debug("Building and compiling the InsuranceAwareAgent Graph")
            
            # 创建状态图
            graph = StateGraph(InsuranceAwareAgentState)
            
            # 添加节点
            graph.add_node("agent", self.agent_node)
            graph.add_node("tool", self.tool_node)
            
            # 添加条件边
            graph.add_conditional_edges(
                "agent",
                self.conditional_edge,
                {
                    "use_pingan_search": "tool",
                    "use_regular_search": "tool"
                }
            )
            
            # 设置入口点
            graph.set_entry_point("agent")
            
            # 编译图
            self.graph = graph.compile()
            logger.info("InsuranceAwareAgent Graph built and compiled successfully")
            
            return self.graph
        except Exception as ex:
            logger.error("Failed to build InsuranceAwareAgent Graph: %s", ex)
            raise


# 注册自定义Agent
from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class InsuranceAwareAgentConfig(FunctionBaseConfig, name="insurance_aware_agent"):
    """
    保险感知Agent配置类
    """
    llm_name: LLMRef = Field(..., description="用于保险感知Agent的LLM模型")
    tool_names: List[str] = Field(default_factory=list, description="提供给Agent的工具列表")
    verbose: bool = Field(default=False, description="设置Agent日志的详细程度")


@register_function(config_type=InsuranceAwareAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def insurance_aware_agent_workflow(config: InsuranceAwareAgentConfig, builder: Builder):
    """
    保险感知Agent工作流实现
    """
    from langchain_core.messages import SystemMessage
    from langgraph.graph.graph import CompiledGraph
    
    # 获取配置的LLM和工具
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    if not tools:
        raise ValueError(f"未为保险感知Agent指定工具")
    
    # 构建Agent图
    graph: CompiledGraph = await InsuranceAwareReActAgent(
        llm=llm,
        tools=tools,
        detailed_logs=config.verbose
    ).build_graph()
    
    async def _insurance_aware_agent(input_message: str) -> str:
        """处理用户输入并返回Agent响应"""
        # 创建初始状态
        state = InsuranceAwareAgentState(
            messages=[SystemMessage(content="你是一个保险领域助手，能够智能识别用户问题类型并选择合适的工具")],
            user_query=input_message
        )
        
        # 执行图
        state = await graph.ainvoke(state)
        
        # 返回最终结果
        return state["messages"][-1].content
    
    try:
        yield FunctionInfo.create(single_fn=_insurance_aware_agent)
    except Exception:
        logger.error("保险感知Agent执行出错")
        raise
    finally:
        logger.info("保险感知Agent执行完毕")