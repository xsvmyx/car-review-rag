from discord import app_commands
import discord

def setup(tree: app_commands.CommandTree, guild_id: int):
    @tree.command(
        name="hello",
        description="Say hello!",
        guild=discord.Object(id=guild_id)
    )
    async def hello(interaction: discord.Interaction):
        await interaction.response.send_message(f"Hello {interaction.user.mention}! 👋")
