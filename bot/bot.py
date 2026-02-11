import discord
from discord import app_commands
from config import TOKEN, GUILD_ID




intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)



from bot.commands import hello,review


hello.setup(tree, GUILD_ID)
review.setup(tree, GUILD_ID)



# --- Listener messages classiques ---
@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.content.lower() == "ping":
        await message.channel.send("Pong!")



# --- Listener on_ready ---
@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")
    # Synchronisation des commandes sur la guild
    await tree.sync(guild=discord.Object(id=GUILD_ID))
    print(f"✅ Commands synced for guild {GUILD_ID}")





# --- Lancer le bot ---
client.run(TOKEN)
