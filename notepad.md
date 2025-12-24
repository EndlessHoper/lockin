
› ok. can you rewrite it to use together ai's hosted gemma model as per the code example below.

  from together import Together

  client = Together()

  response = client.chat.completions.create(
      model="google/gemma-3n-E4B-it",
      messages=[
        {
          "role": "user",
          "content": "What are some fun things to do in New York?"
        }
      ]
  )
  print(response.choices[0].message.content)




  key is tgp_v1_kLnGH6eAJ9TWVttIv54R5rHBwfZMuIT1D3qx_us97yg
  i want you to hardcode it. there is no risk. its my code. its not going anywehre.