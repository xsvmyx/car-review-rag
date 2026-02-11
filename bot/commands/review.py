from discord import app_commands
import discord
from bot.ragUtils.ragService import ask_car_reviews_bot_free_text  # adapte le chemin

def setup(tree: app_commands.CommandTree, guild_id: int):

    @tree.command(
        name="review",
        description="Giving car information based on real user reviews",
        guild=discord.Object(id=guild_id)
    )
    @app_commands.describe(query="Car name or comparison")
    async def review(interaction: discord.Interaction, query: str):
        
        print("COMMAND RECEIVED")
        await interaction.response.defer()
        print("DEFER DONE")

        # Appel de ta fonction RAG
        response = ask_car_reviews_bot_free_text(query)

        await interaction.followup.send(response)