query_string = """
SELECT
  o.Date OrderDate,
  ocs.Date AS TransactionDate,
  o.Id AS OrderId,
  o.ClientCode,
  cl.ManagerId As ManagerId,
  cl.RFMD,
  cl.Segment,
  o.PriceTypeId,
  pt.UnifiedType AS UnifiedPriceType,
  o.PlatformId,
  o.Status,
  pl.Name AS Pratform,
  pl.Region AS Region,
  pl.IsExternal AS IsExternal,
  o.CountryId,
  cnt.NameEn AS OrderCountry,
  o.StateId,
  st.NameEn AS State,
  otr.StoreGroup,
  otr.ProductId,
  pr.NameEn AS Product,
  pr.BrandId AS BrandId,
  br.Name AS Brand,
  pr.IsVirtual AS IsVirtual,
  pr.CategoryId AS CategoryId,
  otr.Quantity AS OrderQty,
  otr.Amount AS OrderAmount,
  ocs.Quantity AS SoldQty,
  ocs.Cost AS SoldCost,
  obp.Price AS BasePrice,
  oup.Price AS UserPrice
FROM
  prosteer.Orders AS o
  LEFT JOIN
  prosteer.Countries AS cnt
  ON
  cnt.Id = o.CountryId
  LEFT JOIN
  prosteer.PriceTypes AS pt
  ON
  pt.Id = o.PriceTypeId
  LEFT JOIN
  prosteer.States AS st
  ON
  st.Id = o.StateId
  LEFT JOIN
  prosteer.OrderTransactions AS otr
  ON
  otr.OrderId = o.Id
  LEFT JOIN
  prosteer.Platforms AS pl
  ON
  o.PlatformId = pl.Id
  LEFT JOIN
  prosteer.Products AS pr
  ON
  otr.ProductId = pr.Id
  LEFT JOIN
  prosteer.Categories AS cat
  ON
  pr.CategoryId = cat.Id
  LEFT JOIN
  prosteer.Clients AS cl
  ON
  o.ClientCode = cl.UserId
  LEFT JOIN
  prosteer.Brands AS br
  ON
  br.Id = pr.BrandId
  LEFT JOIN
  prosteer.OrderCosts AS ocs
  ON
  ocs.OrderId = o.Id
    AND ocs.ProductId = otr.ProductId
  LEFT JOIN
  prosteer.OrderBasePrices AS obp
  ON
  obp.OrderId = o.Id
    AND obp.ProductId = otr.ProductId
  LEFT JOIN
  prosteer.OrderUserPrices AS oup
  ON
  oup.OrderId = o.Id
    AND oup.ProductId = otr.ProductId
WHERE
  o.PlatformId IS NOT NULL
"""
