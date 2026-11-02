"""The same CRM and knowledge base as an MCP server over stdio.

Run with: uv run --extra mcp python mcp_server.py
Here the MCP host owns the loop, so its tool approval is the human gate for the write tools.
"""
from functools import wraps

import crm
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

crm.init_db()
server = MCPServer("support-crm")


def tool(function):
    """Register a tool and pass its error text to the caller, so a model can recover from it."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (LookupError, ValueError) as e:
            raise ToolError(str(e)) from None

    return server.tool()(wrapper)


@tool
def search_kb(query: str) -> list[dict]:
    """Search the store's policies. Each result has a cite value to quote."""
    return crm.search_kb(query)


@tool
def get_customer(email: str) -> dict:
    """Find a customer by email address."""
    return crm.get_customer(email)


@tool
def get_order(customer_id: str, order_id: str) -> dict:
    """Get one of a customer's orders."""
    return crm.get_order(customer_id, order_id)


@tool
def update_address(customer_id: str, address: str) -> dict:
    """Change a customer's address. Needs house number, street and city. Writes data."""
    return crm.update_address(customer_id, address)


@tool
def create_ticket(customer_id: str, subject: str, body: str) -> dict:
    """Open a support ticket. Writes data."""
    return crm.create_ticket(customer_id, subject, body)


@tool
def escalate_to_human(reason: str, summary: str) -> dict:
    """Queue the conversation for a human agent, with a handoff note. Writes data."""
    return crm.escalate_to_human(reason, summary, transcript=[])


if __name__ == "__main__":
    server.run()
